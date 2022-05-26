import os
import concurrent.futures
import json
import argparse
import glob
import importlib  
import sys

import torch

def read_options(argv=None):
    parser = argparse.ArgumentParser(description='Options')    
    #-------------------------------------------------------------------------
    # Data input settings
    parser.add_argument('-visdial_train', default='data/all/visdial_1.0_train_with_vispro_bert_tokenized.json', help='json file containing train split of visdial data')

    parser.add_argument('-visdial_val', default='data/all/visdial_1.0_val_with_vispro_bert_tokenized.json',
                            help='json file containing val split of visdial data')
    parser.add_argument('-visdial_test', default='data/all/visdial_1.0_test_with_vispro_bert_tokenized.json',
                            help='json file containing test split of visdial data')
    parser.add_argument('-visdial_val_ndcg', default='data/all/visdial_1.0_val_dense_annotations.json',
                            help='JSON file with dense annotations')
    parser.add_argument('-visdial_train_ndcg', default='data/all/visdial_1.0_train_dense_annotations.json',
                            help='JSON file with dense annotations')
    parser.add_argument('-visdial_train_dense', default='data/all/visdial_1.0_train_dense_with_vispro_bert_tokenized.json',
                            help='JSON file with dense annotations')

    parser.add_argument('-max_seq_len', default=256, type=int,
                            help='the max len of the input representation of the dialog encoder')
    #-------------------------------------------------------------------------
    # Logging settings

    parser.add_argument('-save_path_train', default='data/processed/visdial_1.0_train_with_vispro_bert_tokenized.json',
                            help='Path to save processed train json')
    parser.add_argument('-save_path_val', default='data/processed/visdial_1.0_val_with_vispro_bert_tokenized.json',
                            help='Path to save val json')
    parser.add_argument('-save_path_test', default='data/processed/visdial_1.0_test_with_vispro_bert_tokenized.json',
                            help='Path to save test json')

    parser.add_argument('-save_path_train_dense_samples', default='data/processed/visdial_1.0_train_dense_with_vispro_bert_tokenized.json',
                            help='Path to save processed train json')
    parser.add_argument('-save_path_val_ndcg', default='data/processed/visdial_1.0_val_dense_annotations.json',
                            help='Path to save processed ndcg data for the val split')
    parser.add_argument('-save_path_train_ndcg', default='data/processed/visdial_1.0_train_dense_annotations.json',
                            help='Path to save processed ndcg data for the train split')

    try:
        parsed = vars(parser.parse_args(args=argv))
    except IOError as msg:
        parser.error(str(msg)) 
    return parsed

if __name__ == "__main__":
    params = read_options() 
    # read all the three splits 

    print(f'Loading {params["visdial_train"]}')
    f = open(params['visdial_train'])
    input_train = json.load(f)
    input_train_data = input_train['data']['dialogs']
    train_questions = input_train['data']['questions']
    train_answers = input_train['data']['answers']
    f.close()

    # read train dense annotations
    print(f'Loading {params["visdial_train_ndcg"]}')
    f = open(params['visdial_train_ndcg'])
    input_train_ndcg = json.load(f)
    f.close()
    print(f'Loading {params["visdial_train_dense"]}')
    f = open(params['visdial_train_dense'])
    input_train_dense = json.load(f)
    input_train_dense_data = input_train_dense['data']['dialogs']
    train_dense_questions = input_train_dense['data']['questions']
    train_dense_answers = input_train_dense['data']['answers']
    f.close()

    print(f'Loading {params["visdial_val"]}')
    f = open(params['visdial_val'])
    input_val = json.load(f)
    input_val_data = input_val['data']['dialogs']
    val_questions = input_val['data']['questions']
    val_answers = input_val['data']['answers'] 
    f.close()

    f = open(params['visdial_val_ndcg'])
    input_val_ndcg = json.load(f)
    f.close()
    
    print(f'Loading {params["visdial_test"]}')
    f = open(params['visdial_test'])
    input_test = json.load(f)
    input_test_data = input_test['data']['dialogs']
    test_questions = input_test['data']['questions']
    test_answers = input_test['data']['answers'] 
    f.close()

    max_seq_len = params["max_seq_len"]
    num_illegal_train = 0
    num_illegal_val = 0
    num_illegal_test = 0
    print('process train')
    # make sure tot_len of training data is smaller than max_len
    # in dataloader, randomly pick negative answer whose tot_len < max_len
    # remove dialogs whose tot_len > max_len
    i = 0
    while i < len(input_train_data):
        cur_dialog = input_train_data[i]['dialog']
        caption = input_train_data[i]['caption']
        tot_len = 22 + len(caption.split(' ')) # account for 21 sep tokens, CLS token and caption
        for rnd in range(len(cur_dialog)):
            tot_len += len(train_answers[cur_dialog[rnd]['answer']].split(' '))
            tot_len += len(train_questions[cur_dialog[rnd]['question']].split(' '))
        if tot_len <= max_seq_len:
            i += 1
        else:
            input_train_data.pop(i)
            num_illegal_train += 1

    train_img_id_processed = [d['image_id'] for d in input_train_data]
    print('pre process dense annotations on train')
    # remove dialogs whose tot_len > max_len
    i = 0
    while i < len(input_train_dense_data):
        remove = False
        if input_train_dense_data[i]['image_id'] in train_img_id_processed:
            assert input_train_dense_data[i]['image_id'] == input_train_ndcg[i]['image_id']
            cur_dialog = input_train_dense_data[i]['dialog']
            img_id = input_train_dense_data[i]['image_id']
            cur_round = input_train_ndcg[i]['round_id'] - 1
            # check if the sample is legal
            caption = input_train_dense_data[i]['caption']
            tot_len = 1 # CLS token
            tot_len += len(caption.split(' ')) + 1
            for rnd in range(cur_round):
                tot_len += len(train_dense_questions[cur_dialog[rnd]['question']].split(' ')) + 1
                if rnd != cur_round - 1:
                    tot_len += len(train_dense_answers[cur_dialog[rnd]['answer']].split(' ')) + 1
            # Unlike train, check each answer option and make sure tot_len < max_len
            # Dense data are used separately from train data
            for option in cur_dialog[cur_round]['answer_options']:
                cur_len = len(train_dense_answers[option].split(' ')) + 1 + tot_len
                if cur_len > max_seq_len:
                    print(f"remove image id {img_id}")
                    remove = True
                    break
        else:
            remove = True

        if remove:
            input_train_dense_data.pop(i)
            input_train_ndcg.pop(i)
        else:
            i += 1
    
    assert len(input_train_ndcg) == len(input_train_dense_data)
    print(f'#dialogs in processed train dense: {len(input_train_dense_data)}')

    print('process val')
    # unlike train, must check if tot_len of all answer options < max_len
    # remove dialogs whose tot_len > max_len
    i = 0
    while i <  len(input_val_data):
        assert input_val_data[i]['image_id'] == input_val_ndcg[i]['image_id']
        remove = False
        cur_dialog = input_val_data[i]['dialog']
        caption = input_val_data[i]['caption']
        tot_len = 1 # CLS token
        tot_len += len(caption.split(' ')) + 1
        for rnd in range(len(cur_dialog)):
            tot_len += len(val_questions[cur_dialog[rnd]['question']].split(' ')) + 1
            for option in cur_dialog[rnd]['answer_options']:
                cur_len = len(val_answers[option].split(' ')) + 1 + tot_len
                if cur_len > max_seq_len:
                    input_val_data.pop(i)
                    input_val_ndcg.pop(i)
                    num_illegal_val += 1
                    remove = True
                    break
            if not remove:
                tot_len += len(val_answers[cur_dialog[rnd]['answer']].split(' ')) + 1
            else:
                break
        if not remove:
            i += 1
    
    i = 0
    print('process test')
    # truncate dialogs whose tot_len > max_len
    while i <  len(input_test_data):
        remove = False
        cur_dialog = input_test_data[i]['dialog']
        input_test_data[i]['round_id'] = len(cur_dialog)
        caption = input_test_data[i]['caption']
        tot_len = 1 # CLS token
        tot_len += len(caption.split(' ')) + 1
        for rnd in range(len(cur_dialog)):
            tot_len += len(test_questions[cur_dialog[rnd]['question']].split(' ')) + 1
            if rnd != len(cur_dialog)-1:
                tot_len += len(test_answers[cur_dialog[rnd]['answer']].split(' ')) + 1
        
        max_len_cur_sample = tot_len

        for option in cur_dialog[-1]['answer_options']:
            cur_len = len(test_answers[option].split(' ')) + 1 + tot_len
            if cur_len > max_seq_len:
                remove = True
                if max_len_cur_sample < cur_len:
                    max_len_cur_sample = cur_len
        if remove:
            # need to process this sample by removing a few rounds
            num_illegal_test += 1
            while max_len_cur_sample > max_seq_len:
                cur_round_len = len(test_questions[cur_dialog[0]['question']].split(' ')) + 1 +  \
                    len(test_answers[cur_dialog[0]['answer']].split(' ')) + 1
                cur_dialog.pop(0)
                max_len_cur_sample -= cur_round_len
        
        i += 1
    '''
    # store processed files
    '''
    print('Writing data')
    if not os.path.exists(os.path.split(params['save_path_train'])[0]):
        os.makedirs(os.path.split(params['save_path_train'])[0])
    with open(params['save_path_train'],'w') as train_out_file:
        json.dump(input_train, train_out_file)

    with open(params['save_path_val'],'w') as val_out_file:
        json.dump(input_val, val_out_file)
    with open(params['save_path_test'],'w') as test_out_file:
        json.dump(input_test, test_out_file)

    with open(params['save_path_train_dense_samples'],'w') as train_dense_out_file:
        json.dump(input_train_dense, train_dense_out_file)

    with open(params['save_path_val_ndcg'],'w') as val_ndcg_out_file:
        json.dump(input_val_ndcg, val_ndcg_out_file)  
    with open(params['save_path_train_ndcg'],'w') as train_ndcg_out_file:
        json.dump(input_train_ndcg, train_ndcg_out_file)

    # spit stats

    print("number of illegal train samples", num_illegal_train)
    print("number of illegal val samples", num_illegal_val)
    print("number of illegal test samples", num_illegal_test)