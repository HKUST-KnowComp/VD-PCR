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

from utils.data_utils import encode_input, encode_input_with_mask, encode_image_input
from utils.image_features_reader import ImageFeaturesH5Reader


class DatasetBase(data.Dataset):

    def __init__(self, config):

        if config['display']:
            log.info('Initializing dataset')

        if config['model_type'] == 'conly' and config['use_embedding'] == 'bert':
            config['dataloader_text_only'] = True

        self.config = config
        self.numDataPoints = {}

        start_time = time.time()
        if not config['dataloader_text_only']:
            self._image_features_reader = ImageFeaturesH5Reader(config['visdial_image_feats'])

        if self.config['training'] or self.config['validating'] or self.config['predicting']:
            split2data = {'train': 'train', 'val': 'val', 'test': 'test'}
        elif self.config['debugging']:
            split2data = {'train': 'val', 'val': 'val', 'test': 'test'}
        elif self.config['visualizing']:
            split2data = {'train': 'train', 'val': 'train', 'test': 'test'}

        filename = f'visdial_{split2data["train"]}'
        if config['train_on_dense']:
            filename += '_dense'
        with open(config[filename]) as f:
            self.visdial_data_train = json.load(f)
            if self.config['predicting'] and self.config['predict_shards_num'] > 1 and self.config['predict_split'] == 'train':
                num_dialogs = len(self.visdial_data_train['data']['dialogs'])
                num_dialog_each_shard = int(np.ceil(num_dialogs / self.config['predict_shards_num']))
                shard_start = num_dialog_each_shard * self.config['predict_shard']
                shard_end = min(num_dialogs + 1, num_dialog_each_shard * (self.config['predict_shard'] + 1))
                print(f'Predicting for dialog {shard_start}-{shard_end - 1} in train split')
                self.visdial_data_train['data']['dialogs'] = self.visdial_data_train['data']['dialogs'][shard_start:shard_end]
            self.numDataPoints['train'] = len(self.visdial_data_train['data']['dialogs'])

        filename = f'visdial_{split2data["val"]}'
        if config['train_on_dense']:
            filename += '_dense'
        with open(config[filename]) as f:
            self.visdial_data_val = json.load(f)
            if self.config['predicting'] and self.config['predict_shards_num'] > 1 and self.config['predict_split'] == 'val':
                num_dialogs = len(self.visdial_data_val['data']['dialogs'])
                num_dialog_each_shard = int(np.ceil(num_dialogs / self.config['predict_shards_num']))
                shard_start = num_dialog_each_shard * self.config['predict_shard']
                shard_end = min(num_dialogs + 1, num_dialog_each_shard * (self.config['predict_shard'] + 1))
                print(f'Predicting for dialog {shard_start}-{shard_end - 1} in val split')
                self.visdial_data_val['data']['dialogs'] = self.visdial_data_val['data']['dialogs'][shard_start:shard_end]
            self.numDataPoints['val'] = len(self.visdial_data_val['data']['dialogs'])
        
        if config['train_on_dense']:
            self.numDataPoints['trainval'] = self.numDataPoints['train'] + self.numDataPoints['val']
        with open(config[f'visdial_{split2data["test"]}']) as f:
            self.visdial_data_test = json.load(f)
            if self.config['predicting'] and self.config['predict_shards_num'] > 1 and self.config['predict_split'] == 'test':
                num_dialogs = len(self.visdial_data_test['data']['dialogs'])
                num_dialog_each_shard = int(np.ceil(num_dialogs / self.config['predict_shards_num']))
                shard_start = num_dialog_each_shard * self.config['predict_shard']
                shard_end = min(num_dialogs + 1, num_dialog_each_shard * (self.config['predict_shard'] + 1))
                print(f'Predicting coreference for dialog {shard_start}-{shard_end - 1} in test split')
                self.visdial_data_test['data']['dialogs'] = self.visdial_data_test['data']['dialogs'][shard_start:shard_end]
            self.numDataPoints['test'] = len(self.visdial_data_test['data']['dialogs'])

        if config['rlv_hst_only']:
            if config['train_on_dense'] and split2data["train"] == 'train':
                self.rlv_hst_train = json.load(open(config[f'rlv_hst_train_dense']))
            else:
                self.rlv_hst_train = json.load(open(config[f'rlv_hst_{split2data["train"]}']))
            self.rlv_hst_val = json.load(open(config[f'rlv_hst_{split2data["val"]}']))
            self.rlv_hst_test = json.load(open(config[f'rlv_hst_{split2data["test"]}']))
        else:
            self.rlv_hst_train = None
            self.rlv_hst_val = None
            self.rlv_hst_test = None

        if config['train_on_dense'] or config['predict_dense_round']:
            with open(config[f'visdial_{split2data["train"]}_dense_annotations']) as f:
                self.visdial_data_train_dense = json.load(f)
        if config['train_on_dense']:
            self.subsets = ['train','val','trainval', 'test']
        else:
            self.subsets = ['train','val','test']
        self.num_options = config["num_options"]
        self.num_options_dense = config["num_options_dense"]
        with open(config[f'visdial_{split2data["val"]}_dense_annotations']) as f:
            self.visdial_data_val_dense = json.load(f)
        self._split = 'train'
        # self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", cache_dir=config['bert_cache_dir'])
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=config['bert_cache_dir'])
        # fetching token indicecs of [CLS] and [SEP]
        tokens = ['[CLS]','[MASK]','[SEP]']
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        self.CLS = indexed_tokens[0]
        self.MASK = indexed_tokens[1]
        self.SEP  = indexed_tokens[2]
        self._max_region_num = 37
        self.predict_each_round = self.config['predicting'] and self.config['predict_each_round']

        self.keys_to_expand = ['image_feat', 'image_loc', 'image_mask', 'image_target', 'image_label']
        self.keys_to_flatten_1d = ['hist_len', 'next_sentence_labels', 'round_id', 'image_id']
        self.keys_to_flatten_2d = ['tokens', 'segments', 'sep_indices', 'mask', 'image_mask', 'image_label', 'input_mask']
        self.keys_to_flatten_3d = ['image_feat', 'image_loc', 'image_target']
        self.keys_other = ['gt_relevance', 'gt_option_inds']
        self.keys_to_list = ['tot_len']
        # for coref
        if config['use_coref']:
            self.genre_to_id = {genre: id_ for id_, genre in enumerate(self.config['id_to_genre'])}
            self.mention_types = ['NP', 'PRP']
            self.keys_to_list.extend(['dialog_info', 'speaker_ids', 'genre_id', 'candidate_starts', 'candidate_ends', 'cand_cluster_ids'])
            self.keys_to_flatten_1d.append('whole_dialog_index_flatten')
            

    def __len__(self):
        return self.numDataPoints[self._split]
    
    @property
    def split(self):
        return self._split

    @split.setter
    def split(self, split):
        assert split in self.subsets
        self._split = split

    def tokens2str(self, seq):
        dialog_sequence = ''
        for sentence in seq:
            for word in sentence:
                dialog_sequence += self.tokenizer._convert_id_to_token(word) + " "
            dialog_sequence += ' </end> '
        dialog_sequence = dialog_sequence.encode('utf8')
        return dialog_sequence

    def pruneRounds(self, context, num_rounds):
        start_segment = 1
        len_context = len(context)
        cur_rounds = (len(context) // 2) + 1
        l_index = 0
        if cur_rounds > num_rounds:
            # caption is not part of the final input
            l_index = len_context - (2 * num_rounds)
            start_segment = 0   
        return context[l_index:], start_segment

    def tokenize_utterance(self, sent, sentences, tot_len, sentence_count, sentence_map, speakers):
        sentences.extend(sent + ['[SEP]'])
        tokenized_sent = self.tokenizer.convert_tokens_to_ids(sent)

        sent_len = len(tokenized_sent)
        tot_len += sent_len + 1 # the additional 1 is for the sep token
        sentence_count += 1
        sentence_map.extend([sentence_count * 2 - 1] * sent_len)
        sentence_map.append(sentence_count * 2) # for [SEP]
        speakers.extend([2] * (sent_len + 1))

        return tokenized_sent, sentences, tot_len, sentence_count, sentence_map, speakers

    def __getitem__(self, index):
        return NotImplementedError

    def collate_fn(self, batch):
        tokens_size = batch[0]['tokens'].size()
        num_rounds, num_samples = tokens_size[0], tokens_size[1]
        sequences_per_image = num_rounds * num_samples
        if self.config['model_type'] == 'joint':
            for i in range(len(batch)):
                batch[i]['whole_dialog_index_flatten'] += i * sequences_per_image

        merged_batch = {key: [d[key] for d in batch] for key in batch[0]}
        out = {}
        for key in merged_batch:
            if key in self.keys_to_list:
                pass
            else:
                merged_batch[key] = torch.stack(merged_batch[key], 0)
                if key in self.keys_to_expand:
                    if len(merged_batch[key].size()) == 3:
                        size0, size1, size2 = merged_batch[key].size()
                        expand_size = (size0, num_rounds, num_samples, size1, size2)
                    elif len(merged_batch[key].size()) == 2:
                        size0, size1 = merged_batch[key].size()
                        expand_size = (size0, num_rounds, num_samples, size1)
                    merged_batch[key] = merged_batch[key].unsqueeze(1).unsqueeze(1).expand(expand_size).contiguous()
                if key in self.keys_to_flatten_1d:
                    merged_batch[key] = merged_batch[key].reshape(-1)
                elif key in self.keys_to_flatten_2d:
                    merged_batch[key] = merged_batch[key].reshape(-1, merged_batch[key].shape[-1])
                elif key in self.keys_to_flatten_3d:
                    merged_batch[key] = merged_batch[key].reshape(-1, merged_batch[key].shape[-2], merged_batch[key].shape[-1])
                else:
                    assert key in self.keys_other, f'unrecognized key in collate_fn: {key}'

            out[key] = merged_batch[key]

        return out
