import json
import os.path as osp
import argparse
import copy
import re

parser = argparse.ArgumentParser(description='correct dense annotation by setting gt answer score to 1')
parser.add_argument('--root_dir', type=str, default='./', 
                    help='root dir')
parser.add_argument('--split', nargs='+', default=['train', 'val'],
                    help='split to process')
parser.add_argument('--data_dir', type=str, default='data', 
                    help='data dir')
parser.add_argument('--processed', action='store_true',
                    help='use processed visdial dialogs')

if __name__ == '__main__':
    args = parser.parse_args()
    if args.processed:
        args.data_dir = osp.join(args.data_dir, 'processed')
    else:
        args.data_dir = osp.join(args.data_dir, 'all')

    for split in args.split:
        filename = osp.join(args.root_dir, args.data_dir, f'visdial_1.0_{split}_dense_annotations.json')
        dense_ann = json.load(open(filename))

        filename = osp.join(args.root_dir, args.data_dir, f'visdial_1.0_{split}_with_vispro_bert_tokenized.json')
        dialogs = json.load(open(filename))['data']['dialogs']
        imid2did = {d['image_id']:i for i, d in enumerate(dialogs)}

        correct_count = 0
        for i in range(len(dense_ann)):
            d = dense_ann[i]
            dialog = dialogs[imid2did[d['image_id']]]
            gt_index = dialog['dialog'][d['round_id'] - 1]['gt_index']
            gt_relevance = d['relevance'] if split == 'train' else d['gt_relevance']
            if abs(gt_relevance[gt_index] - 1) > 1e-2:
                correct_count += 1
                if split == 'train':
                    dense_ann[i]['relevance'][gt_index] = 1
                else:
                    dense_ann[i]['gt_relevance'][gt_index] = 1

        print(f'{correct_count} values are corrected to 1.')
        filename = osp.join(args.root_dir, args.data_dir, f'visdial_1.0_{split}_dense_annotations_corrected.json')
        with open(filename, 'w') as f:
            json.dump(dense_ann, f)
        print(f'Results saved to {filename}')