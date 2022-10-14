#!/usr/bin/env python
import os
import re
import json
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
import glob
import argparse
import os.path as osp
import math

parser = argparse.ArgumentParser(description='extract relevant history from clusters')
parser.add_argument('--split', nargs='+', default=['train_dense', 'val', 'test'],
                    help='split to process')
parser.add_argument('--min', action='store_true',
                    help='Keep the min relevant rounds')
parser.add_argument('--max', action='store_true',
                    help='Keep the max relevant rounds')
parser.add_argument('--include_cap', action='store_true',
                    help='Include caption as relevant for all dialogs')
parser.add_argument('--q_only', action='store_true',
                    help='Do not include any history as relevant')
parser.add_argument('--save_name', type=str, default="",
                    help='save name')
parser.add_argument('--root_dir', type=str, default='./', 
                    help='root dir')
parser.add_argument('--log_dir', type=str, default='logs/conly', 
                    help='log dir')
parser.add_argument('--data_dir', type=str, default='data/all', 
                    help='data dir')
parser.add_argument('--save_dir', type=str, default='data/rlv_hst', 
                    help='save dir')


pronoun_list = ['she', 'her', 'he', 'him', 'them', 'they', 'She', 'Her', 'He', 'Him', 'Them', 'They', 'it', 'It', 'his', 'hers', 'its', 'their', 'theirs', 'His', 'Hers', 'Its', 'Their', 'Theirs']

def sent_id_to_round(sent_id):
    return (sent_id + 1) // 4

def is_answer(sent_id):
    return sent_id > 1 and (sent_id - 1) % 4 == 0

if __name__ == "__main__":
    args = parser.parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    for split in args.split:
        # load data
        print('Loading data')

        split_name = split
        assert split in ['train_dense', 'val', 'test']

        filename = osp.join(args.root_dir, args.data_dir, f'visdial_1.0_{split}_with_vispro_bert_tokenized.json')
        print(f'Loading dialog data from {filename}')
        data = json.load(open(filename))
        cur_questions = data['data']['questions']
        cur_answers = data['data']['answers']

        # deal with each dialog
        print('Extracting relevant history')
        relevant_history_all = OrderedDict()

        for dialog_id, dialog in enumerate(tqdm(data['data']['dialogs'])):
            num_rounds = len(dialog['dialog'])
            image_id = dialog['image_id']
            relevant_history = np.zeros((num_rounds, num_rounds), dtype=np.int32)
                
            if not args.q_only:
                # find sentences and sentence_map
                sents = []

                sent = dialog['caption'].split(' ')
                sents.extend(['[CLS]'] + sent + ['[SEP]'])

                sent_map = [0] # for the CLS token 
                sentence_count = 0
                sentence_count += 1
                sent_map.extend([sentence_count * 2 - 1] * len(sent))
                sent_map.append(sentence_count * 2) # for [SEP]

                for rnd,utterance in enumerate(dialog['dialog']):
                    sent = cur_questions[utterance['question']].split(' ')
                    sents.extend(sent + ['[SEP]'])
                    sentence_count += 1
                    sent_map.extend([sentence_count * 2 - 1] * len(sent))
                    sent_map.append(sentence_count * 2) # for [SEP]

                    if 'answer' in utterance:
                        sent = cur_answers[utterance['answer']].split(' ')
                        sents.extend(sent + ['[SEP]'])
                        sentence_count += 1
                        sent_map.extend([sentence_count * 2 - 1] * len(sent))
                        sent_map.append(sentence_count * 2) # for [SEP]
                    
                # get clusters
                # In the original code, we use PCR annotations for train split and PCR predictions for val, test split.
                # However, to simplify the code here, we directly use the PCR annotation + predictions stored in data
                clusters = [dialog['clusters']] * num_rounds

                # for each cluster
                rounds_to_extract = range(num_rounds)
                for round_id in rounds_to_extract:
                    clusters_round = clusters[round_id]
                    relevant_history_round = relevant_history[round_id]
                    for cluster in clusters_round:
                        # judge history relevance with prp coreference
                        sents_with_NP = set()
                        coref_in_cur_round = False
                        # for each mention, decide if it is a pronoun or a NP
                        for mention in cluster:
                            if mention[0] == mention[1] and sents[mention[0]] in pronoun_list:
                                if is_answer(sent_map[mention[0]]):
                                    # neglect prp in answers
                                    continue
                                # if it is in current round
                                if sent_id_to_round(sent_map[mention[0]]) == round_id + 1:
                                    coref_in_cur_round = True
                            else:
                                assert sent_map[mention[0]] == sent_map[mention[1]], f'mention {mention} of {split} dialog {dialog_id} not in the same sentence'
                                # find its corresponding sentences
                                sents_with_NP.add(sent_id_to_round(sent_map[mention[0]]))
                        if not coref_in_cur_round:
                            # current question does not contain any pronoun or NP of this cluster
                            continue
                        else:
                            sent_with_NP_min = float('inf')
                            sent_with_NP_max = float('-inf')
                            for sent_with_NP in sents_with_NP:
                                if sent_with_NP <= round_id:
                                    if args.min:
                                        sent_with_NP_min = min(sent_with_NP_min, sent_with_NP)
                                    if args.max:
                                        sent_with_NP_max = max(sent_with_NP_max, sent_with_NP)
                                    if (not args.min) and (not args.max): # keep all relevant history
                                        relevant_history_round[sent_with_NP] = 1
                            if args.min:
                                if sent_with_NP_min <= num_rounds:
                                    relevant_history_round[sent_with_NP_min] = 1
                            if args.max:
                                if sent_with_NP_max >= 0:
                                    relevant_history_round[sent_with_NP_max] = 1

            if args.include_cap:
                relevant_history[:, 0] = 1

            relevant_history_all[image_id] = relevant_history.tolist()

        filename = osp.join(args.root_dir, args.save_dir, f'{split}.json')
        if len(args.save_name) > 0:
            filename = filename.replace('.json', f'_{args.save_name}.json')
        with open(filename, 'w') as f:
            json.dump(relevant_history_all, f)
        print(f'relevant_history of {split} split saved to {filename}')
