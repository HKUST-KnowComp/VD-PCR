import torch
from torch.utils import data
import json
import os
import time
import numpy as np
import random
from tqdm import tqdm
import copy
import pyhocon
import glog as log
from collections import OrderedDict
import argparse

import torch.utils.data as tud
from pytorch_transformers.tokenization_bert import BertTokenizer
# from transformers import AutoTokenizer
import sys
# sys.path.append('../')
sys.path.append('./')

from utils.data_utils import encode_input, encode_input_with_mask, encode_image_input
from utils.image_features_reader import ImageFeaturesH5Reader
from dataloader.dataloader_base import DatasetBase


class VisdialDenseDataset(DatasetBase):

    def __init__(self, config):
        super(VisdialDenseDataset, self).__init__(config)


    def __getitem__(self, index):

        # Combining all the dialog rounds with the [SEP] and [CLS] token
        MAX_SEQ_LEN = self.config['max_seq_len']
        cur_data = None
        cur_dense_annotations = None
        if self._split == 'train':
            cur_data = self.visdial_data_train['data']
            cur_dense_annotations = self.visdial_data_train_dense
            if self.config['rlv_hst_only']:
                cur_rlv_hst = self.rlv_hst_train
        elif self._split == 'val':
            cur_data = self.visdial_data_val['data']
            cur_dense_annotations = self.visdial_data_val_dense
            if self.config['rlv_hst_only']:
                cur_rlv_hst = self.rlv_hst_val
        elif self._split == 'trainval':
            if index >= self.numDataPoints['train']:
                cur_data = self.visdial_data_val['data']
                cur_dense_annotations = self.visdial_data_val_dense
                index -= self.numDataPoints['train']
                if self.config['rlv_hst_only']:
                    cur_rlv_hst = self.rlv_hst_val
            else:
                cur_data = self.visdial_data_train['data']
                cur_dense_annotations = self.visdial_data_train_dense
                if self.config['rlv_hst_only']:
                    cur_rlv_hst = self.rlv_hst_train
        elif self._split == 'test':
            cur_data = self.visdial_data_test['data']
            if self.config['rlv_hst_only']:
                cur_rlv_hst = self.rlv_hst_test
        
        # number of options to score on
        num_options = self.num_options_dense
        if self._split == 'test' or self.config['validating'] or self.config['predicting']:
            assert num_options == 100
        else:
            assert num_options >=1 and num_options <=  100
        
        dialog = cur_data['dialogs'][index]
        cur_questions = cur_data['questions']
        cur_answers = cur_data['answers']
        img_id = dialog['image_id']
        if self._split != 'test':
            assert img_id == cur_dense_annotations[index]['image_id']
        if self.config['rlv_hst_only']:
            rlv_hst = cur_rlv_hst[str(img_id)] # [10 for each round, 10 for cap + first 9 round ]

        if self._split == 'test':
            cur_rounds = len(dialog['dialog']) # 1, 2, ..., 10
        else:
            cur_rounds = cur_dense_annotations[index]['round_id'] # 1, 2, ..., 10
        # tot_len = 1

        # caption
        cur_rnd_utterance = []
        include_caption = True
        if self.config['rlv_hst_only']:
            if self.config['rlv_hst_dense_round']:
                if rlv_hst[0] == 0:
                    include_caption = False
            elif rlv_hst[cur_rounds - 1][0] == 0:
                include_caption = False
        if include_caption:
            sent = dialog['caption'].split(' ')
            tokenized_sent = self.tokenizer.convert_tokens_to_ids(sent)
            cur_rnd_utterance.append(tokenized_sent)
            # tot_len += len(sent) + 1

        for rnd,utterance in enumerate(dialog['dialog'][:cur_rounds]):
            if self.config['rlv_hst_only'] and rnd < cur_rounds - 1:
                if self.config['rlv_hst_dense_round']:
                    if rlv_hst[rnd + 1] == 0:
                        continue
                elif rlv_hst[cur_rounds - 1][rnd + 1] == 0:
                    continue
            # question
            sent = cur_questions[utterance['question']].split(' ')
            tokenized_sent = self.tokenizer.convert_tokens_to_ids(sent)
            cur_rnd_utterance.append(tokenized_sent)
            # tot_len += len(sent) + 1

            # answer
            if rnd != cur_rounds - 1:
                sent = cur_answers[utterance['answer']].split(' ')
                tokenized_sent = self.tokenizer.convert_tokens_to_ids(sent)
                cur_rnd_utterance.append(tokenized_sent)
            # tot_len += len(sent) + 1

        if self._split != 'test':
            gt_option = dialog['dialog'][cur_rounds - 1]['gt_index']
            if self.config['training'] or self.config['debugging']:
                # first select gt option id, then choose the first num_options inds
                option_inds = []
                option_inds.append(gt_option) 
                all_inds = list(range(100))
                all_inds.remove(gt_option)
                # debug
                if num_options < 100:
                    random.shuffle(all_inds)
                all_inds = all_inds[:(num_options-1)]
                option_inds.extend(all_inds)
                gt_option = 0
            else:
                option_inds = range(num_options)
            answer_options = [dialog['dialog'][cur_rounds - 1]['answer_options'][k] for k in option_inds]
            if 'relevance' in cur_dense_annotations[index]:
                key = 'relevance'
            else:
                key = 'gt_relevance'
            gt_relevance = torch.Tensor(cur_dense_annotations[index][key])
            gt_relevance = gt_relevance[option_inds]
            assert len(answer_options) == len(option_inds) == num_options
        else:
            answer_options = dialog['dialog'][-1]['answer_options']
            assert len(answer_options) == num_options

        options_all = []
        for answer_option in answer_options:
            cur_option = cur_rnd_utterance.copy()
            cur_option.append(self.tokenizer.convert_tokens_to_ids(cur_answers[answer_option].split(' ')))
            options_all.append(cur_option)
            if not self.config['rlv_hst_only']:
                assert len(cur_option) == 2 * cur_rounds + 1

        tokens_all = []
        mask_all = []
        segments_all = []
        sep_indices_all = []
        hist_len_all = []
        tot_len_debug = []

        for opt_id, option in enumerate(options_all):
            option, start_segment = self.pruneRounds(option, self.config['visdial_tot_rounds'])
            tokens, segments, sep_indices, mask = encode_input(option, start_segment ,self.CLS, 
                    self.SEP, self.MASK ,max_seq_len=MAX_SEQ_LEN, mask_prob=0)

            tokens_all.append(tokens)
            mask_all.append(mask)
            segments_all.append(segments)
            sep_indices_all.append(sep_indices)
            hist_len_all.append(torch.LongTensor([len(option)-1]))

            len_tokens = sum(len(s) for s in option)
            tot_len_debug.append(len_tokens + len(option) + 1)

        tokens_all = torch.cat(tokens_all,0)
        mask_all = torch.cat(mask_all,0)
        segments_all = torch.cat(segments_all, 0)
        sep_indices_all = torch.cat(sep_indices_all, 0)
        hist_len_all = torch.cat(hist_len_all,0)
               
        item = {}

        item['tokens'] = tokens_all.unsqueeze(0) # [1, num_options, max_len]
        item['segments'] = segments_all.unsqueeze(0)
        item['sep_indices'] = sep_indices_all.unsqueeze(0)
        item['mask'] = mask_all.unsqueeze(0)
        item['hist_len'] = hist_len_all.unsqueeze(0)

        # add dense annotation fields
        if self._split != 'test':
            item['gt_relevance'] = gt_relevance # [num_options]
            item['gt_option_inds'] = torch.LongTensor([gt_option])

            # add next sentence labels for training with the nsp loss as well
            nsp_labels = torch.ones(*tokens_all.unsqueeze(0).shape[:-1]).long()
            nsp_labels[:,gt_option] = 0
            item['next_sentence_labels'] = nsp_labels

            item['round_id'] = torch.LongTensor([cur_rounds])
        else:
            if 'round_id' in dialog:
                item['round_id'] = torch.LongTensor([dialog['round_id']])
            else:
                item['round_id'] = torch.LongTensor([cur_rounds])

        # get image features
        if not self.config['dataloader_text_only']:
            features, num_boxes, boxes, _ , image_target = self._image_features_reader[img_id]
            features, spatials, image_mask, image_target, image_label = encode_image_input(features, num_boxes, boxes, image_target, max_regions=self._max_region_num, mask_prob=0)
        else:
            features = spatials = image_mask = image_target = image_label = torch.tensor([0])
        item['image_feat'] = features
        item['image_loc'] = spatials
        item['image_mask'] = image_mask
        item['image_id'] = torch.LongTensor([img_id])

        item['tot_len'] = tot_len_debug

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

    # os.chdir('../')
    config = pyhocon.ConfigFactory.parse_file("config/vonly.conf")[args.model]
    config['training'] = args.mode == 'train'
    config['validating'] = args.mode == 'eval'
    config['debugging'] = args.mode == 'debug'
    config['predicting'] = args.mode == 'predict'
    config['display'] = True
    config['model_type'] = 'vonly'
    config['num_options_dense'] = 100

    dataset = VisdialDenseDataset(config)

    # for split in ['train', 'val', 'trainval']:
    # for split in ['train', 'val', 'test']:
    for split in ['trainval']:
        dataset.split = split
        log.info(f'#{split} examples: {len(dataset)}')

        # data_loader = tud.DataLoader(
        #             dataset=dataset,
        #             batch_size=2,
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


        data_loader = tud.DataLoader(
                    dataset=dataset,
                    batch_size=16,
                    shuffle=False,
                    collate_fn=dataset.collate_fn,
                    num_workers=4
                )

        max_len = 0
        tot_len = OrderedDict()
        for batch in tqdm(data_loader):
            for i, l in enumerate(batch['tot_len']):
                tot_len[str(batch['image_id'][i].item())] = l
                max_len = max(max_len, max(l))

        # json.dump(tot_len, open(f'data/tot_len_rlv_only_{split}.json', 'w'))
        print(f'max_len of all inputs of {split}: {max_len}')