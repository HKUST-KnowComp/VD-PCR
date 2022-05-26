# coding: utf-8
import os
import os.path as osp
import json
import sys
import numpy as np
import argparse
from tqdm import tqdm

import torch
import torch.utils.data as tud

sys.path.append('./')

from models.visual_dialog_encoder import VisualDialogEncoder
from dataloader.dataloader_visdial_coref_only import VisdialPrpOnlyDataset
from utils.init_utils import initialize_from_env
from utils.data_utils import sequence_mask

parser = argparse.ArgumentParser(description='compare bert attention with coreference')
parser.add_argument('--ckpt_path', type=str, default='checkpoints-release/basemodel', 
                    help='checkpoint to analyze')
parser.add_argument('--num_selected_heads', type=int, default=5,
                    help='#selected coreference related heads among all heads')
parser.add_argument('--model', type=str, default='conly/MB-pseudo_eval',
                    help='model name to train or test')
parser.add_argument('--mode', type=str, default='eval',
                    help='train, eval, predict or debug')
parser.add_argument('--split', type=str, default='train', choices=['train', 'val'],
                    help='split to analyze')

NUM_HEADS = 12
NUM_LAYERS = 12
def get_mention(clusters):
    mentions = []
    for cluster in clusters:
        for mention in cluster:
            mentions.append(mention)

    mention_tokens = [' '.join(sentences[m[0]:m[1]+1]) for m in mentions]
    
    return mentions, mention_tokens

def get_mention_att(att, mentions):

    num_mentions = len(mentions)
    att_mention = np.zeros((num_mentions, num_mentions))
    att_token = []
    for mention in mentions:
        att_t = np.sum(att[mention[0]:mention[1]+1], axis=0, keepdims=True)
        att_token.append(att_t)
    att_token = np.concatenate(att_token, axis=0)
    for m_id, mention in enumerate(mentions):
        att_m = np.sum(att_token[:, mention[0]:mention[1]+1], axis=1)
        att_mention[:, m_id] = att_m

    return att_mention

prp_in_NP_list = ['his', 'her', 'their', 'its']
def remove_prp_in_NP(clusters, sentences):
    clusters_new = []
    for cluster in clusters:
        cluster_new = []
        for mention in cluster:
            if mention[1] > mention[0] and sentences[mention[0]] in prp_in_NP_list:
                cluster_new.append([mention[0]+1, mention[1]])
            else:
                cluster_new.append(mention)
        clusters_new.append(cluster_new)
    return clusters_new

if __name__ == "__main__":
    args = parser.parse_args()

    model_type, model_name = args.model.split('/')
    config = initialize_from_env(model_name, args.mode, model_type)
    config['training'] = args.mode == 'train'
    config['validating'] = args.mode == 'eval'
    config['debugging'] = args.mode == 'debug'
    config['predicting'] = args.mode == 'predict'
    config['visualizing'] = args.mode == 'visualize'

    config['display'] = True
    config['num_workers'] = 4
    config['device'] = 'cuda:0'

    # load the model
    map_location = {'cuda:0': config['device']}
    model_to_load = torch.load(args.ckpt_path, map_location=map_location)
    model_to_load = model_to_load['model_state_dict']

    model = VisualDialogEncoder(config['model_config'], config['device'], 
                                use_apex=False, 
                                cache_dir=config['bert_cache_dir'])
    model = model.to(config['device'])

    model_dict = model.state_dict()
    model_to_load = {k: v for k, v in model_to_load.items() if k in model_dict}
    model.load_state_dict(model_to_load)
    print(("number of keys transferred: %d" % len(model_to_load)))

    del model_to_load, model_dict

    model.eval()

    # load data from vispro only
    dataset = VisdialPrpOnlyDataset(config)
    dataset.split = args.split

    item_count = 0
    margins_all = np.zeros([NUM_LAYERS, NUM_HEADS])
    for item_id in tqdm(range(len(dataset))):

        item = dataset[item_id]
        clusters = item['dialog_info'][0]
        if len(clusters) <= 1:
            continue

        item_count += 1

        batch = dataset.collate_fn([item])
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(config['device'])

        # get embedding and forward visdial
        tokens = batch['tokens']
        segments = batch['segments']
        sep_indices = batch['sep_indices']
        mask = batch['mask']
        hist_len = batch['hist_len']
        image_feat = batch['image_feat']
        image_loc = batch['image_loc']
        image_mask = batch['image_mask']

        sequence_lengths = torch.gather(sep_indices, 1, hist_len.view(-1, 1)) + 1
        sequence_lengths = sequence_lengths.squeeze(1)
        attention_mask_lm_nsp = sequence_mask(sequence_lengths, max_len=tokens.shape[1])
        sep_len = hist_len + 1

        with torch.no_grad():
            _, _, _, _, all_attention_mask = model.bert_pretrained(
                tokens,
                image_feat,
                image_loc,
                sep_indices=sep_indices,
                sep_len=sep_len,
                token_type_ids=segments,
                masked_lm_labels=mask,
                attention_mask=attention_mask_lm_nsp,
                image_attention_mask=image_mask,
                output_all_attention_masks=True
            )

        all_attention_mask_t = all_attention_mask[0]
        assert len(all_attention_mask_t) == NUM_LAYERS

        # compare attention with coreference information
        clusters, image_id, pronoun_info, sentences = batch['dialog_info'][0]
        clusters = remove_prp_in_NP(clusters, sentences)
        # remove cluster with only one mention
        i = 0
        while i < len(clusters):
            if len(clusters[i]) <= 1:
                clusters.pop(i)
            else:
                i += 1
        assert len(clusters) > 0

        mentions, mention_tokens = get_mention(clusters)
        mention_count = 0
        clusters_count = []
        for cluster in clusters:
            cluster_count = []
            for mention in cluster:
                cluster_count.append(mention_count)
                mention_count += 1
            clusters_count.append(cluster_count)
        num_clusters = len(clusters)
        num_tokens = len(sentences)

        margins = []
        for layer_id in range(NUM_LAYERS):
            att_layer = all_attention_mask_t[layer_id]
            att_layer = att_layer.squeeze(0)[:, :num_tokens, :num_tokens].cpu().data.numpy()
            margins_layer = []
            for head_id in range(att_layer.shape[0]):
                att_mention = get_mention_att(att_layer[head_id], mentions)
                att_mention -= np.diag(np.diag(att_mention)) # remove self-attention

                # only consider attention on annotated mentions and maximize the margin between one cluster and others
                att_cluster = np.zeros((num_clusters, num_clusters))
                for cluster_id in range(num_clusters):
                    num_mention_cur_cluster = len(clusters_count[cluster_id])
                    for other_cluster_id in range(cluster_id, num_clusters):
                        num_mention_other_cluster = len(clusters_count[other_cluster_id])
                        att_sum = np.sum(att_mention[clusters_count[cluster_id]][:, clusters_count[other_cluster_id]])
                        if cluster_id == other_cluster_id:
                            att_mean = att_sum / (num_mention_cur_cluster * (num_mention_other_cluster - 1))
                        else:
                            att_mean = att_sum / (num_mention_cur_cluster * num_mention_other_cluster)
                        att_cluster[cluster_id, other_cluster_id] = att_mean
                        att_cluster[other_cluster_id, cluster_id] = att_mean
                margin_head = (2 * np.sum(np.diag(att_cluster)) - np.sum(att_cluster)) / num_clusters
                margins_layer.append(margin_head)

            margins.append(margins_layer)

        margins_all += np.array(margins)

    margins_all /= item_count
    print(f'Compare attention of {item_count} dialogs which have more than 2 clusters')
    print('Attention of the last layer:')
    ranks = np.argsort(margins_all[-1])[::-1]
    for i in range(NUM_HEADS):
        print(f'{i}. head {ranks[i]} margin {margins_all[-1][ranks[i]]:.4f}')

    print('Attention of all layers:')
    margins_all = np.ravel(margins_all)
    ranks = np.argsort(margins_all)[::-1]
    for i in range(NUM_LAYERS):
        print(f'{i}. layer {ranks[i] // NUM_HEADS} head {ranks[i] % NUM_HEADS} margin {margins_all[ranks[i]]:.4f}')

    # write to config
    heads_for_coref = []
    for i in range(args.num_selected_heads):
        layer = int(ranks[i] // NUM_HEADS)
        head = int(ranks[i] % NUM_HEADS)
        for head_for_coref in heads_for_coref:
            if head_for_coref['layer'] == layer:
                head_for_coref['heads'].append(head)
                break
        else:
            heads_for_coref.append({'layer': layer, 'heads': [head]})

    bert_config = json.load(open('config/bert_base_6layer_6conect.json'))
    bert_config['heads_for_coref'] = heads_for_coref
    config_filename = 'config/bert_base_6layer_6conect_coref_heads.json'
    with open(config_filename, 'w') as f:
        json.dump(bert_config, f, indent=2)
    print(f'Top {args.num_selected_heads} coreference related heads saved to {config_filename}')