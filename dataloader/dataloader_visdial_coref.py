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

import torch.utils.data as tud
from pytorch_transformers.tokenization_bert import BertTokenizer
# from transformers import AutoTokenizer

import sys
sys.path.append('..')
from utils.data_utils import encode_input, encode_input_with_mask, encode_image_input
from utils.image_features_reader import ImageFeaturesH5Reader
from dataloader.dataloader_base import DatasetBase


class VisdialPrpDataset(DatasetBase):

    def __init__(self, config):
        super(VisdialPrpDataset, self).__init__(config)

    def __getitem__(self, index):

        # Combining all the dialog rounds with the [SEP] and [CLS] token
        MAX_SEQ_LEN = self.config['max_seq_len']
        cur_data = None
        if self._split == 'train':
            cur_data = self.visdial_data_train['data']
        elif self._split == 'val':
            cur_data = self.visdial_data_val['data']
        else:
            cur_data = self.visdial_data_test['data']
        
        # number of options to score on
        num_options = self.num_options
        assert num_options > 1 and num_options <= 100
        num_dialog_rounds = 10
        
        dialog = cur_data['dialogs'][index]
        cur_questions = cur_data['questions']
        cur_answers = cur_data['answers']
        img_id = dialog['image_id']

        if self._split == 'train':

            # caption
            sent = dialog['caption'].split(' ')
            sentences = ['[CLS]']
            tot_len = 1 # for the CLS token 
            sentence_map = [0] # for the CLS token 
            sentence_count = 0
            speakers = [0]

            tokenized_sent, sentences, tot_len, sentence_count, sentence_map, speakers = \
                self.tokenize_utterance(sent, sentences, tot_len, sentence_count, sentence_map, speakers)

            utterances = [[tokenized_sent]]
            utterances_random = [[tokenized_sent]]

            for rnd,utterance in enumerate(dialog['dialog']):
                cur_rnd_utterance = utterances[-1].copy()
                cur_rnd_utterance_random = utterances[-1].copy()
                
                # question
                sent = cur_questions[utterance['question']].split(' ')
                tokenized_sent, sentences, tot_len, sentence_count, sentence_map, speakers = \
                    self.tokenize_utterance(sent, sentences, tot_len, sentence_count, sentence_map, speakers)

                cur_rnd_utterance.append(tokenized_sent)
                cur_rnd_utterance_random.append(tokenized_sent)

                # answer
                sent = cur_answers[utterance['answer']].split(' ')
                tokenized_sent, sentences, tot_len, sentence_count, sentence_map, speakers = \
                    self.tokenize_utterance(sent, sentences, tot_len, sentence_count, sentence_map, speakers)
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

                    while len(all_inds):
                        option_ind = random.choice(all_inds)
                        tokenized_random_utterance = self.tokenizer.convert_tokens_to_ids(cur_answers[utterance['answer_options'][option_ind]].split(' '))
                        # the 1 here is for the sep token at the end of each utterance
                        if(MAX_SEQ_LEN >= (tot_len + len(tokenized_random_utterance) + 1)):
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
            assert tot_len <= MAX_SEQ_LEN, f'{self._split} {index} tot_len > max_seq_len'

            tokens_all = []
            mask_all = []
            segments_all = []
            sep_indices_all = []
            next_labels_all = []
            hist_len_all = []

            if self.config['coref_only']:
                pos_rounds = [num_dialog_rounds - 1]
                neg_rounds = []
            else:
                # randomly pick several rounds to train
                pos_rounds = sorted(random.sample(range(num_dialog_rounds), self.config['sequences_per_image'] // 2), reverse=True)
                neg_rounds = sorted(random.sample(range(num_dialog_rounds), self.config['sequences_per_image'] // 2), reverse=True)

            tokens_all_rnd = []
            mask_all_rnd = []
            segments_all_rnd = []
            sep_indices_all_rnd = []
            next_labels_all_rnd = []
            hist_len_all_rnd = []

            for j in pos_rounds:

                context = utterances[j]
                context, start_segment = self.pruneRounds(context, self.config['visdial_tot_rounds'])
                if j == pos_rounds[0]: # dialog with positive label and max rounds
                    tokens, segments, sep_indices, mask, input_mask = encode_input_with_mask(context, start_segment, self.CLS,
                     self.SEP, self.MASK, max_seq_len=MAX_SEQ_LEN, mask_prob=self.config["mask_prob"])
                else:
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

            if len(neg_rounds) > 0:
                tokens_all_rnd = []
                mask_all_rnd = []
                segments_all_rnd = []
                sep_indices_all_rnd = []
                next_labels_all_rnd = []
                hist_len_all_rnd = []

                for j in neg_rounds:

                    negative_samples = utterances_random[j]
                    for context_random in negative_samples:
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
            input_mask_all = torch.LongTensor(input_mask) # [max_len]
                   
            item = {}

            item['tokens'] = tokens_all
            item['segments'] = segments_all
            item['sep_indices'] = sep_indices_all
            item['mask'] = mask_all
            item['next_sentence_labels'] = next_labels_all
            item['hist_len'] = hist_len_all
            item['input_mask'] = input_mask_all
            item['whole_dialog_index_flatten'] = torch.LongTensor([0])

            # for coref truncate
            max_pos_round = pos_rounds[0]
            if max_pos_round == num_dialog_rounds - 1:
                max_pos_tokens = len(sentences)
            else:
                max_pos_tokens = sentence_map.index((max_pos_round + 1) * 4 + 3)
                sentences = sentences[:max_pos_tokens]
                sentence_map = sentence_map[:max_pos_tokens]

            # get image features
            if not self.config['dataloader_text_only']:
                features, num_boxes, boxes, _ , image_target = self._image_features_reader[img_id]
                features, spatials, image_mask, image_target, image_label = encode_image_input(features, num_boxes, boxes, image_target, max_regions=self._max_region_num)
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
            sentence_map = [0] # for the CLS token 
            sentence_count = 0
            speakers = [0]

            tokenized_sent, sentences, tot_len, sentence_count, sentence_map, speakers = \
                self.tokenize_utterance(sent, sentences, tot_len, sentence_count, sentence_map, speakers)
            utterances = [[tokenized_sent]]

            for rnd,utterance in enumerate(dialog['dialog']):
                cur_rnd_utterance = utterances[-1].copy()
                
                # question
                sent = cur_questions[utterance['question']].split(' ')
                tokenized_sent, sentences, tot_len, sentence_count, sentence_map, speakers = \
                    self.tokenize_utterance(sent, sentences, tot_len, sentence_count, sentence_map, speakers)

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
                    if 'relevance' in self.visdial_data_val_dense[index]:
                        gt_relevance = torch.Tensor(self.visdial_data_val_dense[index]['relevance'])
                    else:
                        gt_relevance = torch.Tensor(self.visdial_data_val_dense[index]['gt_relevance'])
                    # shuffle based on new indices
                    gt_relevance = gt_relevance[torch.LongTensor(option_inds)]
                for answer_option in answer_options:
                    cur_rnd_cur_option = cur_rnd_utterance.copy()
                    cur_rnd_cur_option.append(self.tokenizer.convert_tokens_to_ids(cur_answers[answer_option].split(' ')))
                    cur_rnd_options.append(cur_rnd_cur_option)

                # answer
                sent = cur_answers[utterance['answer']].split(' ')
                tokenized_sent, sentences, tot_len, sentence_count, sentence_map, speakers = \
                    self.tokenize_utterance(sent, sentences, tot_len, sentence_count, sentence_map, speakers)
                cur_rnd_utterance.append(tokenized_sent)

                utterances.append(cur_rnd_utterance)
                options_all.append(cur_rnd_options)

            # encode the input and create batch x 10 x 100 * max_len arrays (batch x num_rounds x num_options)            
            tokens_all = []
            mask_all = []
            segments_all = []
            sep_indices_all = []
            hist_len_all = []

            for rnd, cur_rnd_options in enumerate(options_all):

                tokens_all_rnd = []
                mask_all_rnd = []
                segments_all_rnd = []
                sep_indices_all_rnd = []
                hist_len_all_rnd = []

                for j, cur_rnd_option in enumerate(cur_rnd_options):

                    cur_rnd_option, start_segment = self.pruneRounds(cur_rnd_option, self.config['visdial_tot_rounds'])
                    if rnd == len(options_all) - 1 and j == 0: # gt dialog
                        tokens, segments, sep_indices, mask, input_mask = encode_input_with_mask(cur_rnd_option, start_segment, self.CLS,
                         self.SEP, self.MASK, max_seq_len=MAX_SEQ_LEN, mask_prob=0)
                    else:
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
            input_mask_all = torch.LongTensor(input_mask) # [max_len]
                   
            item = {}

            item['tokens'] = tokens_all
            item['segments'] = segments_all
            item['sep_indices'] = sep_indices_all
            item['mask'] = mask_all
            item['hist_len'] = hist_len_all
            item['input_mask'] = input_mask_all
            item['whole_dialog_index_flatten'] = torch.LongTensor([num_options * (num_dialog_rounds - 1)])

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
            if self.config['eval_coref_on_test']:

                # caption
                sent = dialog['caption'].split(' ')
                sentences = ['[CLS]']
                tot_len = 1 # for the CLS token 
                sentence_map = [0] # for the CLS token 
                sentence_count = 0
                speakers = [0]

                tokenized_sent, sentences, tot_len, sentence_count, sentence_map, speakers = \
                    self.tokenize_utterance(sent, sentences, tot_len, sentence_count, sentence_map, speakers)
                cur_rnd_utterance = [tokenized_sent]

                for rnd,utterance in enumerate(dialog['dialog']):
                    # question
                    sent = cur_questions[utterance['question']].split(' ')
                    tokenized_sent, sentences, tot_len, sentence_count, sentence_map, speakers = \
                        self.tokenize_utterance(sent, sentences, tot_len, sentence_count, sentence_map, speakers)

                    cur_rnd_utterance.append(tokenized_sent)

                    if rnd != len(dialog['dialog'])-1:
                        # answer
                        sent = cur_answers[utterance['answer']].split(' ')
                        tokenized_sent, sentences, tot_len, sentence_count, sentence_map, speakers = \
                            self.tokenize_utterance(sent, sentences, tot_len, sentence_count, sentence_map, speakers)
                        cur_rnd_utterance.append(tokenized_sent)

                cur_rnd_utterance, start_segment = self.pruneRounds(cur_rnd_utterance, self.config['visdial_tot_rounds'])
                tokens, segments, sep_indices, mask, input_mask = encode_input_with_mask(cur_rnd_utterance, start_segment ,self.CLS, 
                self.SEP, self.MASK ,max_seq_len=MAX_SEQ_LEN, mask_prob=0)

                tokens_all = [tokens]
                mask_all = [mask]
                segments_all = [segments]
                sep_indices_all = [sep_indices]
                hist_len_all = [torch.LongTensor([len(cur_rnd_utterance)-1])]
                    
                tokens_all = torch.cat(tokens_all,0) # [1, max_len]
                mask_all = torch.cat(mask_all,0)
                segments_all = torch.cat(segments_all, 0)
                sep_indices_all = torch.cat(sep_indices_all, 0)
                hist_len_all = torch.cat(hist_len_all,0)
                input_mask_all = torch.LongTensor(input_mask) # [max_len]

                item = {}
                item['tokens'] = tokens_all.unsqueeze(0) # [1, 1, max_len]
                item['segments'] = segments_all.unsqueeze(0)
                item['sep_indices'] = sep_indices_all.unsqueeze(0)
                item['mask'] = mask_all.unsqueeze(0)
                item['hist_len'] = hist_len_all.unsqueeze(0)
                item['input_mask'] = input_mask_all
                item['whole_dialog_index_flatten'] = torch.LongTensor([0])

                # for coref truncate
                max_pos_tokens = None

            elif self.config['eval_visdial_on_test']:

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
                
                for j, option in enumerate(options_all):
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

            item['round_id'] = torch.LongTensor([dialog['round_id']])
        
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

        # for coreference
        genre_id = torch.as_tensor([self.genre_to_id['dl']])
        clusters = copy.deepcopy(dialog['clusters'])
        num_words = len(sentences)
        if self._split in ['val', 'test'] and 'pronoun_info' in dialog:
            pronoun_info = copy.deepcopy(dialog['pronoun_info'])
        else:
            pronoun_info = None

        # keep mentions < max_pos_tokens
        if max_pos_tokens is not None:
            clusters_truncated = []
            for cluster in clusters:
                cluster_truncated = []
                for mention in cluster:
                    if mention[1] < max_pos_tokens:
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

        # [num_words, max_span_width]
        candidate_starts = torch.arange(num_words).view(-1, 1).repeat(1, self.config['max_span_width'])
        # [num_words, max_span_width]
        cand_cluster_ids = torch.zeros_like(candidate_starts)

        if gold_mentions:
            gold_end_offsets = gold_ends - gold_starts
            gold_mention_mask = gold_end_offsets < self.config['max_span_width']
            filtered_gold_starts = gold_starts[gold_mention_mask]
            filtered_gold_end_offsets = gold_end_offsets[gold_mention_mask]
            filtered_gold_cluster_ids = gold_cluster_ids[gold_mention_mask]
            cand_cluster_ids[filtered_gold_starts, filtered_gold_end_offsets] = filtered_gold_cluster_ids

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

        item['dialog_info'] = [
            clusters,
            dialog['image_id'], 
            pronoun_info, 
            sentences
        ]

        item['speaker_ids'] = torch.tensor(speakers, dtype=torch.long)
        item['genre_id'] = genre_id
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
    config = pyhocon.ConfigFactory.parse_file("config/joint.conf")[args.model]
    config['training'] = args.mode == 'train'
    config['validating'] = args.mode == 'eval'
    config['debugging'] = args.mode == 'debug'
    config['predicting'] = args.mode == 'predict'
    config['display'] = True
    config['model_type'] = 'joint'

    dataset = VisdialPrpDataset(config)

    # for split in ['train', 'val', 'test']:
    for split in ['val']:
        dataset.split = split
        log.info(f'#{split} examples: {len(dataset)}')

        data_loader = tud.DataLoader(
                    dataset=dataset,
                    batch_size=2,
                    shuffle=False,
                    collate_fn=dataset.collate_fn,
                    num_workers=0
                )

        count = 0
        for batch in tqdm(data_loader):
            count += 1
        #     if count >= 10:
        #         break
        # print(f'Check dataloader for 10 batches.')

        # for i in range(len(dataset)):
        #     item = dataset[i]

        print(f'Check whole dataloader.')
