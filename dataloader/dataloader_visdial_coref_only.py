import torch
from torch.utils import data
import json
import os
import time
# from transformers import AutoTokenizer
import numpy as np
import random
from tqdm import tqdm
import copy
import argparse
import pyhocon
import glog as log

from pytorch_transformers.tokenization_bert import BertTokenizer
import torch.utils.data as tud

from utils.data_utils import list2tensorpad, encode_input_with_mask, encode_image_input
from utils.image_features_reader import ImageFeaturesH5Reader
from dataloader.dataloader_base import DatasetBase


class VisdialPrpOnlyDataset(DatasetBase):

    def __init__(self, config):
        super(VisdialPrpOnlyDataset, self).__init__(config)
        if self.config['predict_dense_round']:
            self.imid2denseid_train = {d['image_id']: i for i, d in enumerate(self.visdial_data_train_dense)}
            self.imid2denseid_val = {d['image_id']: i for i, d in enumerate(self.visdial_data_val_dense)}

    def __len__(self):
        return self.numDataPoints[self._split]
    
    @property
    def split(self):
        return self._split

    @split.setter
    def split(self, split):
        assert split in self.subsets
        self._split = split

    def __getitem__(self, index):

        # Combining all the dialog rounds with the [SEP] and [CLS] token
        MAX_SEQ_LEN = self.config['max_seq_len']
        cur_data = None
        if self._split == 'train':
            cur_data = self.visdial_data_train['data']
            if self.config['predict_dense_round']:
                cur_dense_annotations = self.visdial_data_train_dense
                cur_imid2denseid = self.imid2denseid_train
        elif self._split == 'val':
            cur_data = self.visdial_data_val['data']
            if self.config['predict_dense_round']:
                cur_dense_annotations = self.visdial_data_val_dense
                cur_imid2denseid = self.imid2denseid_val
        else:
            cur_data = self.visdial_data_test['data']
        
        # number of options to score on
        num_options = self.num_options
        assert num_options > 1 and num_options <= 100
        
        dialog = cur_data['dialogs'][index]
        cur_questions = cur_data['questions']
        cur_answers = cur_data['answers']
        img_id = dialog['image_id']
        sentences = []
        total_rounds = len(dialog['dialog'])

        utterances = []
        # utterances_random = []
        sent = dialog['caption'].split(' ')
        sentences.extend(['[CLS]'] + sent + ['[SEP]'])
        tokenized_caption = self.tokenizer.convert_tokens_to_ids(sent) 
        utterances.append([tokenized_caption])
        # utterances_random.append([tokenized_caption])

        tot_len = 1 # for the CLS token 
        sentence_map = [0] # for the CLS token 
        sentence_count = 0
        tot_len += len(tokenized_caption) + 1 # for the sep tokens which follows the caption
        sentence_count += 1
        sentence_map.extend([sentence_count * 2 - 1] * len(tokenized_caption))
        sentence_map.append(sentence_count * 2) # for [SEP]
        speakers = [0] * tot_len

        for rnd,utterance in enumerate(dialog['dialog']):
            cur_rnd_utterance = utterances[-1].copy()
            
            sent = cur_questions[utterance['question']].split(' ')
            sentences.extend(sent + ['[SEP]'])
            tokenized_question = self.tokenizer.convert_tokens_to_ids(sent)
            cur_rnd_utterance.append(tokenized_question)

            question_len = len(tokenized_question)
            tot_len += question_len + 1 # the additional 1 is for the sep token
            sentence_count += 1
            sentence_map.extend([sentence_count * 2 - 1] * question_len)
            sentence_map.append(sentence_count * 2) # for [SEP]
            speakers.extend([1] * (question_len + 1))

            if 'answer' in utterance:
                sent = cur_answers[utterance['answer']].split(' ')
                sentences.extend(sent + ['[SEP]'])
                tokenized_answer = self.tokenizer.convert_tokens_to_ids(sent)
                cur_rnd_utterance.append(tokenized_answer)

                answer_len = len(tokenized_answer)
                tot_len += answer_len + 1 # the additional 1 is for the sep token
                sentence_count += 1
                sentence_map.extend([sentence_count * 2 - 1] * answer_len)
                sentence_map.append(sentence_count * 2) # for [SEP]
                speakers.extend([2] * (answer_len + 1))

            utterances.append(cur_rnd_utterance)

        # removing the caption in the beginning
        utterances = utterances[1:]

        tokens_all = []
        mask_all = []
        segments_all = []
        sep_indices_all = []
        hist_len_all = []
        input_mask_all = []

        if self.config['predict_dense_round']:
            # for predict_dense_round, only take the round with dense ann
            if self._split in ['train', 'val']:
                dense_round_id = cur_dense_annotations[cur_imid2denseid[img_id]]['round_id'] - 1
                utterances = [utterances[dense_round_id][:-1]]
            else:
                dense_round_id = len(dialog['dialog']) - 1
                utterances = [utterances[dense_round_id]]
            tot_len = sum(len(s) for s in utterances[0]) + len(utterances[0]) + 1
        elif self.config['predict_each_round']:
            # for predict_each_round, take each round as input and drop answer
            if self._split in ['train', 'val']:
                # drop answer
                for rnd in range(len(utterances)):
                    utterances[rnd] = utterances[rnd][:-1]
                tot_len = sum(len(s) for s in utterances[-1]) + len(utterances[-1]) + 1
        elif self.config['train_each_round']:
            # for training, randomly sample 1 round;
            # for evaluation, take all rounds as input
            if self._split == 'train':
                if self.config['drop_last_answer']:
                    utterances = [random.choice(utterances)[:-1]]
                else:
                    utterances = [random.choice(utterances)]
            elif self._split == 'val':
                if self.config['drop_last_answer']:
                    utterances = [u[:-1] for u in utterances]
            tot_len = sum(len(s) for s in utterances[-1]) + len(utterances[-1]) + 1
        else:
            # for coreference training, only take the whole dialog as output
            utterances = [utterances[-1]]

        assert tot_len <= MAX_SEQ_LEN, f'{self._split} {index} tot_len > max_seq_len'

        for j,context in enumerate(utterances):
            context, start_segment = self.pruneRounds(context, self.config['visdial_tot_rounds'])
            tokens, segments, sep_indices, mask, input_mask = encode_input_with_mask(context, start_segment, self.CLS,
             self.SEP, self.MASK, max_seq_len=MAX_SEQ_LEN, mask_prob=self.config["mask_prob"])

            tokens_all.append(tokens) # [1, max_len]
            mask_all.append(mask)
            segments_all.append(segments)
            sep_indices_all.append(sep_indices)
            input_mask_all.append(input_mask)
            hist_len_all.append(torch.LongTensor([len(context)-1])) # [1]

        tokens_all = torch.cat(tokens_all,0).unsqueeze(1) # [1, 10, max_len]
        mask_all = torch.cat(mask_all,0).unsqueeze(1)
        segments_all = torch.cat(segments_all, 0).unsqueeze(1)
        sep_indices_all = torch.cat(sep_indices_all, 0).unsqueeze(1)
        input_mask_all = torch.cat(input_mask_all,0).unsqueeze(1) # [1, 10, max_len]
        hist_len_all = torch.cat(hist_len_all,0) # [10]
               
        item = {}

        item['tokens'] = tokens_all
        item['segments'] = segments_all
        item['sep_indices'] = sep_indices_all
        item['mask'] = mask_all
        item['hist_len'] = hist_len_all
        item['input_mask'] = input_mask_all
        
        # get image features
        if not self.config['dataloader_text_only']:
            features, num_boxes, boxes, _ , image_target = self._image_features_reader[img_id]
            features, spatials, image_mask, image_target, image_label = encode_image_input(features, num_boxes, boxes, image_target, max_regions=self._max_region_num, mask_prob=self.config["mask_prob"])
        else:
            features = spatials = image_mask = image_target = image_label = torch.tensor([0])
        item['image_feat'] = features
        item['image_loc'] = spatials
        item['image_mask'] = image_mask
        item['image_target'] = image_target
        item['image_label'] = image_label
        item['image_id'] = torch.LongTensor([img_id])

        # for coreference
        item['speaker_ids'] = torch.tensor(speakers, dtype=torch.long)
        genre_id = torch.as_tensor([self.genre_to_id['dl']])
        item['genre_id'] = genre_id

        num_words = tot_len
        # [num_words, max_span_width]
        candidate_starts = torch.arange(num_words).view(-1, 1).repeat(1, self.config['max_span_width'])
        # [num_words, max_span_width]
        cand_cluster_ids = torch.zeros_like(candidate_starts)

        if 'clusters' in dialog:
            clusters = copy.deepcopy(dialog['clusters'])
            # keep mentions < tot_len
            if self.config['train_each_round'] and self._split in ['train', 'val']:
                clusters_truncated = []
                for cluster in clusters:
                    cluster_truncated = []
                    for mention in cluster:
                        if mention[1] < tot_len:
                            cluster_truncated.append(mention)
                    if len(cluster_truncated) > 0:
                        clusters_truncated.append(cluster_truncated)
                clusters = clusters_truncated

            gold_mentions = sorted(
                tuple(mention) for cluster in clusters
                for mention in cluster
            )

            gold_mention_to_id = {
                mention: id_
                for id_, mention in enumerate(gold_mentions)
            }

            gold_starts, gold_ends = map(
                # np.array,
                torch.as_tensor,
                zip(*gold_mentions) if gold_mentions else ([], [])
            )

            gold_cluster_ids = torch.zeros(len(gold_mentions), dtype=torch.long)

            for cluster_id, cluster in enumerate(clusters):
                for mention in cluster:
                    # leave cluster_id of 0 for dummy
                    gold_cluster_ids[gold_mention_to_id[tuple(mention)]] = cluster_id + 1

            if gold_mentions:
                gold_end_offsets = gold_ends - gold_starts
                gold_mention_mask = gold_end_offsets < self.config['max_span_width']
                filtered_gold_starts = gold_starts[gold_mention_mask]
                filtered_gold_end_offsets = gold_end_offsets[gold_mention_mask]
                filtered_gold_cluster_ids = gold_cluster_ids[gold_mention_mask]
                cand_cluster_ids[filtered_gold_starts, filtered_gold_end_offsets] = filtered_gold_cluster_ids
        else:
            clusters = None

        # [num_words * max_span_width]
        candidate_ends = (candidate_starts + torch.arange(self.config['max_span_width']).view(1, -1)).view(-1)

        sentence_indices = torch.tensor(sentence_map)

        # remove cands with cand_ends >= num_words
        # [num_words * max_span_width]
        candidate_starts = candidate_starts.view(-1)
        # [num_words * max_span_width]
        cand_cluster_ids = cand_cluster_ids.view(-1)
        # [num_words * max_span_width]
        cand_mask = candidate_ends < num_words
        # [cand_num]
        candidate_starts = candidate_starts[cand_mask]
        # [cand_num]
        candidate_ends = candidate_ends[cand_mask]
        # [cand_num]
        cand_cluster_ids = cand_cluster_ids[cand_mask]

        # remove cands whose start and end not in the same sentences
        # [cand_num]
        cand_start_sent_idxes = sentence_indices[candidate_starts]
        # [cand_num]
        cand_end_sent_idxes = sentence_indices[candidate_ends]
        # [cand_num]
        cand_mask = (cand_start_sent_idxes == cand_end_sent_idxes)
        # [cand_num]
        candidate_starts = candidate_starts[cand_mask]
        # [cand_num]
        candidate_ends = candidate_ends[cand_mask]
        # [cand_num]
        cand_cluster_ids = cand_cluster_ids[cand_mask]

        if 'pronoun_info' in dialog:
            pronoun_info = copy.deepcopy(dialog['pronoun_info'])
        else:
            pronoun_info = None

        # deal with dialog info
        if self.config['train_each_round'] and self._split != 'train':
            dialog_info = []
            max_sent_map = max(sentence_map)
            clusters = copy.deepcopy(dialog['clusters'])
            for rnd in range(total_rounds):
                if self.config['drop_last_answer']:
                    max_token = sentence_map.index(min((rnd + 1) * 4 + 1, max_sent_map))
                else:
                    max_token = sentence_map.index(min((rnd + 1) * 4 + 3, max_sent_map))
                # for cluster
                clusters_rnd = []
                for cluster in clusters:
                    cluster_rnd = []
                    for mention in cluster:
                        if mention[1] < max_token:
                            cluster_rnd.append(mention)
                    if len(cluster_rnd) > 0:
                        clusters_rnd.append(cluster_rnd)
                # for pronoun_info
                pronoun_info_rnd = []
                for prp in pronoun_info:
                    if prp['current_pronoun'][0] >= max_token:
                        continue
                    correct_NPs_rnd = copy.deepcopy(prp['correct_NPs'])
                    i = 0
                    while i < len(correct_NPs_rnd):
                        if correct_NPs_rnd[i][1] >= max_token:
                            correct_NPs_rnd.pop(i)
                        else:
                            i += 1
                    cand_NPs_rnd = copy.deepcopy(prp['candidate_NPs'])
                    i = 0
                    while i < len(cand_NPs_rnd):
                        if cand_NPs_rnd[i][1] >= max_token:
                            cand_NPs_rnd.pop(i)
                        else:
                            i += 1
                    prp_rnd = {
                        "current_pronoun": prp['current_pronoun'],
                        "correct_NPs": correct_NPs_rnd,
                        "candidate_NPs": cand_NPs_rnd,
                        "reference_type": prp['reference_type']
                    }
                    pronoun_info_rnd.append(prp_rnd)
                sentences_rnd = sentences[:max_token]
                dialog_info.append([clusters_rnd, dialog['image_id'],
                                   pronoun_info_rnd, sentences_rnd])
            item['dialog_info'] = dialog_info
        else:
            item['dialog_info'] = [
                clusters,
                dialog['image_id'], 
                pronoun_info, 
                sentences
            ]

        # deal with candidate starts & ends
        if self.config['predict_each_round'] or (self.config['train_each_round'] and self._split != 'train'):
            candidate_starts_rnd = []
            candidate_ends_rnd = []
            cand_cluster_ids_rnd = []
            max_sent_map = max(sentence_map)
            for rnd in range(total_rounds):
                if self.config['predict_each_round'] or self.config['drop_last_answer']:
                    max_token = sentence_map.index(min((rnd + 1) * 4 + 1, max_sent_map))
                else:
                    max_token = sentence_map.index(min((rnd + 1) * 4 + 3, max_sent_map))
                candidate_range = candidate_ends < max_token
                candidate_starts_rnd.append(candidate_starts[candidate_range])
                candidate_ends_rnd.append(candidate_ends[candidate_range])
                cand_cluster_ids_rnd.append(cand_cluster_ids[candidate_range])
            item['candidate_starts'] = candidate_starts_rnd
            item['candidate_ends'] = candidate_ends_rnd
            item['cand_cluster_ids'] = cand_cluster_ids_rnd

            # copy features
            item['speaker_ids'] = [item['speaker_ids']] * total_rounds
            item['genre_id'] = [item['genre_id']] * total_rounds
        else:
            item['candidate_starts'] = candidate_starts
            item['candidate_ends'] = candidate_ends
            item['cand_cluster_ids'] = cand_cluster_ids

        return item


def parse_args():
    parser = argparse.ArgumentParser(description='debug dataloader')
    parser.add_argument('--mode', type=str,
                        help='train, eval or debug')
    parser.add_argument('--model', type=str,
                        help='model name to train or test')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    os.chdir('../')
    config = pyhocon.ConfigFactory.parse_file("config/conly.conf")[args.model]
    config['training'] = args.mode == 'train'
    config['validating'] = args.mode == 'eval'
    config['debugging'] = args.mode == 'debug'
    config['predicting'] = args.mode == 'predict'
    config['display'] = True
    config['model_type'] = 'conly'

    dataset = VisdialPrpOnlyDataset(config)

    # for split in ['train', 'val', 'test']:
    for split in ['val']:
        dataset.split = split
        log.info(f'#{split} examples: {len(dataset)}')

        # data_loader = tud.DataLoader(
        #             dataset=dataset,
        #             batch_size=1,
        #             shuffle=False,
        #             collate_fn=dataset.collate_fn,
        #             num_workers=0
        #         )

        # count = 0
        # for batch in data_loader:
        #     count += 1
        #     if count >= 10:
        #         break
        # print(f'Check dataloader for {count} batches.')

        for i in range(len(dataset)):
            print(i)
            item = dataset[i]