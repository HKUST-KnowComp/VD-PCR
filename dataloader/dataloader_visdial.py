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
import argparse
from collections import OrderedDict

import torch.utils.data as tud
from pytorch_transformers.tokenization_bert import BertTokenizer
# from transformers import AutoTokenizer
import sys
# sys.path.append('../')
sys.path.append('./')

from utils.data_utils import encode_input, encode_input_with_mask, encode_image_input
from utils.image_features_reader import ImageFeaturesH5Reader
from dataloader.dataloader_base import DatasetBase


class VisdialDataset(DatasetBase):

    def __init__(self, config):
        super(VisdialDataset, self).__init__(config)


    def __getitem__(self, index):

        # Combining all the dialog rounds with the [SEP] and [CLS] token
        MAX_SEQ_LEN = self.config['max_seq_len']
        cur_data = None
        if self._split == 'train':
            cur_data = self.visdial_data_train['data']
            cur_rlv_hst = self.rlv_hst_train
        elif self._split == 'val':
            cur_data = self.visdial_data_val['data']
            cur_rlv_hst = self.rlv_hst_val
        else:
            cur_data = self.visdial_data_test['data']
            cur_rlv_hst = self.rlv_hst_test
        
        # number of options to score on
        num_options = self.num_options
        assert num_options > 1 and num_options <= 100
        num_dialog_rounds = 10
        
        dialog = cur_data['dialogs'][index]
        cur_questions = cur_data['questions']
        cur_answers = cur_data['answers']
        img_id = dialog['image_id']
        if self.config['rlv_hst_only']:
            rlv_hst = cur_rlv_hst[str(img_id)]

        if self._split == 'train':

            # caption
            sent = dialog['caption'].split(' ')
            tot_len = 1 # for the CLS token 

            tokenized_sent = self.tokenizer.convert_tokens_to_ids(sent)
            sent_len = len(tokenized_sent)
            tot_len += sent_len + 1 # the additional 1 is for the sep token

            utterances = [[tokenized_sent]]
            utterances_random = [[tokenized_sent]]

            for rnd,utterance in enumerate(dialog['dialog']):
                cur_rnd_utterance = utterances[-1].copy()
                cur_rnd_utterance_random = utterances[-1].copy()
                
                # question
                sent = cur_questions[utterance['question']].split(' ')
                tokenized_sent = self.tokenizer.convert_tokens_to_ids(sent)
                sent_len = len(tokenized_sent)
                tot_len += sent_len + 1 # the additional 1 is for the sep token

                cur_rnd_utterance.append(tokenized_sent)
                cur_rnd_utterance_random.append(tokenized_sent)

                # answer
                sent = cur_answers[utterance['answer']].split(' ')
                tokenized_sent = self.tokenizer.convert_tokens_to_ids(sent)
                sent_len = len(tokenized_sent)
                tot_len += sent_len + 1 # the additional 1 is for the sep token
                
                cur_rnd_utterance.append(tokenized_sent)

                utterances.append(cur_rnd_utterance)

                # randomly select one random utterance in that round
                num_inds = len(utterance['answer_options'])
                gt_option_ind = utterance['gt_index']

                negative_samples = []

                for _ in range(self.config["num_negative_samples"]):

                    all_inds = list(range(100))
                    all_inds.remove(gt_option_ind)
                    all_inds = all_inds[:(num_options-1)]
                    tokenized_random_utterance = None
                    option_ind = None

                    if self.config['rlv_hst_only']:
                        tot_len_rlv = 1
                        for dialog_round in range(rnd + 1): # dialog_round: 0 for caption, 1...10 for dialog
                            if rlv_hst[rnd][dialog_round] == 1:
                                if dialog_round == 0:
                                    tot_len_rlv += len(cur_rnd_utterance_random[0]) + 1
                                else:
                                    tot_len_rlv += len(cur_rnd_utterance_random[dialog_round * 2 - 1]) + 1
                                    tot_len_rlv += len(cur_rnd_utterance_random[dialog_round * 2]) + 1
                        tot_len_rlv += len(cur_rnd_utterance_random[-1]) + 1
                        tot_len_cur_neg = tot_len_rlv
                    else:
                        tot_len_cur_neg = tot_len

                    while len(all_inds):
                        option_ind = random.choice(all_inds)
                        tokenized_random_utterance = self.tokenizer.convert_tokens_to_ids(cur_answers[utterance['answer_options'][option_ind]].split(' '))

                        # the 1 here is for the sep token at the end of each utterance
                        if(MAX_SEQ_LEN >= (tot_len_cur_neg + len(tokenized_random_utterance) + 1)):
                            break
                        else:
                            all_inds.remove(option_ind)
                    if len(all_inds) == 0:
                        # all the options exceed the max len. Truncate the last utterance in this case.
                        tokenized_random_utterance = tokenized_random_utterance[:len(tokenized_sent)]
                    t = cur_rnd_utterance_random.copy()
                    t.append(tokenized_random_utterance)
                    negative_samples.append(t)

                utterances_random.append(negative_samples)

            # removing the caption in the beginning
            utterances = utterances[1:]
            utterances_random = utterances_random[1:]
            assert len(utterances) == len(utterances_random) == num_dialog_rounds
            if not self.config['rlv_hst_only']:
                assert tot_len <= MAX_SEQ_LEN, f'{self._split} {index} tot_len > max_seq_len'

            tokens_all = []
            mask_all = []
            segments_all = []
            sep_indices_all = []
            next_labels_all = []
            hist_len_all = []

            # randomly pick several rounds to train
            pos_rounds = sorted(random.sample(range(num_dialog_rounds), self.config['sequences_per_image'] // 2), reverse=True)
            neg_rounds = sorted(random.sample(range(num_dialog_rounds), self.config['sequences_per_image'] // 2), reverse=True)

            tokens_all_rnd = []
            mask_all_rnd = []
            segments_all_rnd = []
            sep_indices_all_rnd = []
            next_labels_all_rnd = []
            hist_len_all_rnd = []
            tot_len_debug = []

            for j in pos_rounds:
                context = utterances[j]

                if self.config['rlv_hst_only']:
                    sents_to_pop = []
                    for dialog_round in range(j + 1): # dialog_round: 0 for caption, 1...10 for dialog
                        if rlv_hst[j][dialog_round] == 0:
                            if dialog_round == 0:
                                sents_to_pop.append(0)
                            else:
                                sents_to_pop.append(dialog_round * 2 - 1)
                                sents_to_pop.append(dialog_round * 2)

                    pop_count = 0
                    for sent_id in sents_to_pop:
                        context.pop(sent_id - pop_count)
                        pop_count += 1

                    len_tokens = sum(len(s) for s in context) + len(context) + 1
                    tot_len_debug.append(len_tokens)
                    assert len_tokens <= MAX_SEQ_LEN, f'{self._split} dialog.{index} pos_round.{j} tot_len > max_seq_len'

                context, start_segment = self.pruneRounds(context, self.config['visdial_tot_rounds'])
                tokens, segments, sep_indices, mask = encode_input(context, start_segment, self.CLS,
                    self.SEP, self.MASK, max_seq_len=MAX_SEQ_LEN, mask_prob=self.config["mask_prob"])
                tokens_all_rnd.append(tokens)
                mask_all_rnd.append(mask)
                sep_indices_all_rnd.append(sep_indices)
                next_labels_all_rnd.append(torch.LongTensor([0]))
                segments_all_rnd.append(segments)
                hist_len_all_rnd.append(torch.LongTensor([len(context)-1]))

            tokens_all.append(torch.cat(tokens_all_rnd,0).unsqueeze(0))
            mask_all.append(torch.cat(mask_all_rnd,0).unsqueeze(0))
            segments_all.append(torch.cat(segments_all_rnd, 0).unsqueeze(0))
            sep_indices_all.append(torch.cat(sep_indices_all_rnd, 0).unsqueeze(0))
            next_labels_all.append(torch.cat(next_labels_all_rnd, 0).unsqueeze(0))
            hist_len_all.append(torch.cat(hist_len_all_rnd,0).unsqueeze(0))

            tokens_all_rnd = []
            mask_all_rnd = []
            segments_all_rnd = []
            sep_indices_all_rnd = []
            next_labels_all_rnd = []
            hist_len_all_rnd = []

            for j in neg_rounds:

                negative_samples = utterances_random[j]
                for context_random in negative_samples:
                    if self.config['rlv_hst_only']:
                        sents_to_pop = []
                        for dialog_round in range(j + 1): # dialog_round: 0 for caption, 1...10 for dialog
                            if rlv_hst[j][dialog_round] == 0:
                                if dialog_round == 0:
                                    sents_to_pop.append(0)
                                else:
                                    sents_to_pop.append(dialog_round * 2 - 1)
                                    sents_to_pop.append(dialog_round * 2)

                        pop_count = 0
                        for sent_id in sents_to_pop:
                            context_random.pop(sent_id - pop_count)
                            pop_count += 1

                        len_tokens = sum(len(s) for s in context_random) + len(context_random) + 1
                        tot_len_debug.append(len_tokens)
                        assert len_tokens <= MAX_SEQ_LEN, f'{self._split} dialog.{index} neg_round.{j} tot_len > max_seq_len'

                    context_random, start_segment = self.pruneRounds(context_random, self.config['visdial_tot_rounds'])
                    tokens_random, segments_random, sep_indices_random, mask_random = encode_input(context_random, start_segment, self.CLS, 
                    self.SEP, self.MASK, max_seq_len=MAX_SEQ_LEN, mask_prob=self.config["mask_prob"])
                    tokens_all_rnd.append(tokens_random)
                    mask_all_rnd.append(mask_random)
                    sep_indices_all_rnd.append(sep_indices_random)
                    next_labels_all_rnd.append(torch.LongTensor([1]))
                    segments_all_rnd.append(segments_random)
                    hist_len_all_rnd.append(torch.LongTensor([len(context_random)-1]))

            tokens_all.append(torch.cat(tokens_all_rnd,0).unsqueeze(0))
            mask_all.append(torch.cat(mask_all_rnd,0).unsqueeze(0))
            segments_all.append(torch.cat(segments_all_rnd, 0).unsqueeze(0))
            sep_indices_all.append(torch.cat(sep_indices_all_rnd, 0).unsqueeze(0))
            next_labels_all.append(torch.cat(next_labels_all_rnd, 0).unsqueeze(0))
            hist_len_all.append(torch.cat(hist_len_all_rnd,0).unsqueeze(0))

            tokens_all = torch.cat(tokens_all,0) # [2, num_pos, max_len]
            mask_all = torch.cat(mask_all,0)
            segments_all = torch.cat(segments_all, 0)
            sep_indices_all = torch.cat(sep_indices_all, 0)
            next_labels_all = torch.cat(next_labels_all, 0)
            hist_len_all = torch.cat(hist_len_all,0)

            item = {}

            item['tokens'] = tokens_all
            item['segments'] = segments_all
            item['sep_indices'] = sep_indices_all
            item['mask'] = mask_all
            item['next_sentence_labels'] = next_labels_all
            item['hist_len'] = hist_len_all

            # get image features
            if not self.config['dataloader_text_only']:
                features, num_boxes, boxes, _ , image_target = self._image_features_reader[img_id]
                features, spatials, image_mask, image_target, image_label = encode_image_input(features, num_boxes, boxes, image_target, max_regions=self._max_region_num, mask_prob=self.config['image_mask_prob'])
            else:
                features = spatials = image_mask = image_target = image_label = torch.tensor([0])

        elif self._split == 'val':
            gt_relevance = None
            gt_option_inds = []
            options_all = []

            # caption
            sent = dialog['caption'].split(' ')
            sentences = ['[CLS]']
            tot_len = 1 # for the CLS token 

            tokenized_sent = self.tokenizer.convert_tokens_to_ids(sent)
            sent_len = len(tokenized_sent)
            tot_len += sent_len + 1 # the additional 1 is for the sep token
            utterances = [[tokenized_sent]]

            for rnd,utterance in enumerate(dialog['dialog']):
                cur_rnd_utterance = utterances[-1].copy()
                
                # question
                sent = cur_questions[utterance['question']].split(' ')
                tokenized_sent = self.tokenizer.convert_tokens_to_ids(sent)
                sent_len = len(tokenized_sent)
                tot_len += sent_len + 1 # the additional 1 is for the sep token

                cur_rnd_utterance.append(tokenized_sent)

                # current round
                gt_option_ind = utterance['gt_index']
                # first select gt option id, then choose the first num_options inds
                option_inds = []
                option_inds.append(gt_option_ind) 
                all_inds = list(range(100))
                all_inds.remove(gt_option_ind)
                all_inds = all_inds[:(num_options-1)]
                option_inds.extend(all_inds)
                gt_option_inds.append(0)
                cur_rnd_options = []
                answer_options = [utterance['answer_options'][k] for k in option_inds]
                assert len(answer_options) == len(option_inds) == num_options
                assert answer_options[0] == utterance['answer']

                # for evaluation of all options and dense relevance
                if rnd == self.visdial_data_val_dense[index]['round_id'] - 1:
                    # only 1 round has gt_relevance for each example
                    gt_relevance = torch.Tensor(self.visdial_data_val_dense[index]['gt_relevance'])
                    # shuffle based on new indices
                    gt_relevance = gt_relevance[torch.LongTensor(option_inds)]
                for answer_option in answer_options:
                    cur_rnd_cur_option = cur_rnd_utterance.copy()
                    cur_rnd_cur_option.append(self.tokenizer.convert_tokens_to_ids(cur_answers[answer_option].split(' ')))
                    cur_rnd_options.append(cur_rnd_cur_option)

                # answer
                sent = cur_answers[utterance['answer']].split(' ')
                tokenized_sent = self.tokenizer.convert_tokens_to_ids(sent)
                sent_len = len(tokenized_sent)
                tot_len += sent_len + 1 # the additional 1 is for the sep token
                cur_rnd_utterance.append(tokenized_sent)

                utterances.append(cur_rnd_utterance)
                options_all.append(cur_rnd_options)

            # encode the input and create batch x 10 x 100 * max_len arrays (batch x num_rounds x num_options)            
            tokens_all = []
            mask_all = []
            segments_all = []
            sep_indices_all = []
            hist_len_all = []
            tot_len_debug = []

            for rnd, cur_rnd_options in enumerate(options_all):

                tokens_all_rnd = []
                mask_all_rnd = []
                segments_all_rnd = []
                sep_indices_all_rnd = []
                hist_len_all_rnd = []

                if self.config['rlv_hst_only']:
                    sents_to_pop = []
                    for dialog_round in range(rnd + 1): # dialog_round: 0 for caption, 1...10 for dialog
                        if rlv_hst[rnd][dialog_round] == 0:
                            if dialog_round == 0:
                                sents_to_pop.append(0)
                            else:
                                sents_to_pop.append(dialog_round * 2 - 1)
                                sents_to_pop.append(dialog_round * 2)

                for j, cur_rnd_option in enumerate(cur_rnd_options):

                    if self.config['rlv_hst_only']:
                        pop_count = 0
                        for sent_id in sents_to_pop:
                            cur_rnd_option.pop(sent_id - pop_count)
                            pop_count += 1
                        len_tokens = sum(len(s) for s in cur_rnd_option)
                        tot_len_debug.append(len_tokens + len(cur_rnd_option) + 1)

                    cur_rnd_option, start_segment = self.pruneRounds(cur_rnd_option, self.config['visdial_tot_rounds'])
                    tokens, segments, sep_indices, mask = encode_input(cur_rnd_option, start_segment,self.CLS, 
                        self.SEP, self.MASK ,max_seq_len=MAX_SEQ_LEN, mask_prob=0)

                    tokens_all_rnd.append(tokens)
                    mask_all_rnd.append(mask)
                    segments_all_rnd.append(segments)
                    sep_indices_all_rnd.append(sep_indices)
                    hist_len_all_rnd.append(torch.LongTensor([len(cur_rnd_option)-1]))

                tokens_all.append(torch.cat(tokens_all_rnd,0).unsqueeze(0))
                mask_all.append(torch.cat(mask_all_rnd,0).unsqueeze(0))
                segments_all.append(torch.cat(segments_all_rnd,0).unsqueeze(0))
                sep_indices_all.append(torch.cat(sep_indices_all_rnd,0).unsqueeze(0))
                hist_len_all.append(torch.cat(hist_len_all_rnd,0).unsqueeze(0))

            tokens_all = torch.cat(tokens_all,0) # [10, 100, max_len]
            mask_all = torch.cat(mask_all,0)
            segments_all = torch.cat(segments_all, 0)
            sep_indices_all = torch.cat(sep_indices_all, 0)
            hist_len_all = torch.cat(hist_len_all,0)

            item = {}

            item['tokens'] = tokens_all
            item['segments'] = segments_all
            item['sep_indices'] = sep_indices_all
            item['mask'] = mask_all
            item['hist_len'] = hist_len_all

            item['gt_option_inds'] = torch.LongTensor(gt_option_inds)            

            # return dense annotation data as well
            item['round_id'] = torch.LongTensor([self.visdial_data_val_dense[index]['round_id']])
            item['gt_relevance'] = gt_relevance

            # for coref truncate
            max_pos_tokens = None

            # get image features
            if not self.config['dataloader_text_only']:
                features, num_boxes, boxes, _ , image_target = self._image_features_reader[img_id]
                features, spatials, image_mask, image_target, image_label = encode_image_input(features, num_boxes, boxes, image_target, max_regions=self._max_region_num, mask_prob=0)
            else:
                features = spatials = image_mask = image_target = image_label = torch.tensor([0])

        elif self.split == 'test':
            assert num_options == 100
            cur_rnd_utterance = [self.tokenizer.convert_tokens_to_ids(dialog['caption'].split(' '))]
            options_all = []
            for rnd,utterance in enumerate(dialog['dialog']):
                cur_rnd_utterance.append(self.tokenizer.convert_tokens_to_ids(cur_questions[utterance['question']].split(' ')))
                if rnd != len(dialog['dialog'])-1:
                    cur_rnd_utterance.append(self.tokenizer.convert_tokens_to_ids(cur_answers[utterance['answer']].split(' ')))
            for answer_option in dialog['dialog'][-1]['answer_options']:
                cur_option = cur_rnd_utterance.copy()
                cur_option.append(self.tokenizer.convert_tokens_to_ids(cur_answers[answer_option].split(' ')))
                options_all.append(cur_option)

            tokens_all = []
            mask_all = []
            segments_all = []
            sep_indices_all = []
            hist_len_all = []
            # debug
            tot_len_debug = []
            
            for j, option in enumerate(options_all):
                # debug
                len_tokens = sum(len(s) for s in option)
                tot_len_debug.append(len_tokens + len(option) + 1)

                option, start_segment = self.pruneRounds(option, self.config['visdial_tot_rounds'])
                # print("option: {} {}".format(j, tokens2str(option)))
                tokens, segments, sep_indices, mask = encode_input(option, start_segment ,self.CLS, 
                self.SEP, self.MASK ,max_seq_len=MAX_SEQ_LEN, mask_prob=0)

                tokens_all.append(tokens)
                mask_all.append(mask)
                segments_all.append(segments)
                sep_indices_all.append(sep_indices)
                hist_len_all.append(torch.LongTensor([len(option)-1]))
                
            tokens_all = torch.cat(tokens_all,0)
            mask_all = torch.cat(mask_all,0)
            segments_all = torch.cat(segments_all, 0)
            sep_indices_all = torch.cat(sep_indices_all, 0)
            hist_len_all = torch.cat(hist_len_all,0)

            item = {}
            item['tokens'] = tokens_all.unsqueeze(0)
            item['segments'] = segments_all.unsqueeze(0)
            item['sep_indices'] = sep_indices_all.unsqueeze(0)
            item['mask'] = mask_all.unsqueeze(0)
            item['hist_len'] = hist_len_all.unsqueeze(0)
            if 'round_id' in dialog:
                item['round_id'] = torch.LongTensor([dialog['round_id']]) # processed
            else:
                item['round_id'] = torch.LongTensor([len(dialog['dialog'])]) # all

            # get image features
            if not self.config['dataloader_text_only']:
                features, num_boxes, boxes, _ , image_target = self._image_features_reader[img_id]
                features, spatials, image_mask, image_target, image_label = encode_image_input(features, num_boxes, boxes, image_target, max_regions=self._max_region_num, mask_prob=0)
            else:
                features = spatials = image_mask = image_target = image_label = torch.tensor([0])
        item['image_feat'] = features
        item['image_loc'] = spatials
        item['image_mask'] = image_mask
        item['image_target'] = image_target
        item['image_label'] = image_label
        item['image_id'] = torch.LongTensor([img_id])

        # debug
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

    dataset = VisdialDataset(config)

    for split in ['train', 'val', 'test']:
    # for split in ['val']:
        dataset.split = split
        log.info(f'#{split} examples: {len(dataset)}')

        # data_loader = tud.DataLoader(
        #             dataset=dataset,
        #             batch_size=16,
        #             shuffle=False,
        #             collate_fn=dataset.collate_fn,
        #             num_workers=4
        #         )

        # count = 0
        # for batch in data_loader:
        #     count += 1
        #     if count >= 10:
        #         break
        # print(f'Check dataloader for 10 batches.')

        data_loader = tud.DataLoader(
                    dataset=dataset,
                    batch_size=128,
                    shuffle=False,
                    collate_fn=dataset.collate_fn,
                    num_workers=10
                )

        # data_loader = tud.DataLoader(
        #             dataset=dataset,
        #             batch_size=2,
        #             shuffle=False,
        #             collate_fn=dataset.collate_fn,
        #             num_workers=0
        #         )

        max_len = 0
        tot_len = OrderedDict()
        for batch in tqdm(data_loader):
            for i, l in enumerate(batch['tot_len']):
                tot_len[str(batch['image_id'][i].item())] = l
                max_len = max(max_len, max(l))

        json.dump(tot_len, open(f'data/tot_len_rlv_only_{split}_not_dense.json', 'w'))
        print(f'max_len of all inputs of {split}: {max_len}')