import json
import os.path as osp
import argparse
import copy

parser = argparse.ArgumentParser(description='keep VisDial dialogs with VisPro annotations only')
parser.add_argument('--split', nargs='+', default=['val', 'test'],
                    help='split to process')
parser.add_argument('--root_dir', type=str, default='./', 
                    help='root dir')
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
        print(f'Processing {split} split')

        # Load data of VisDial
        data = json.load(open(osp.join(args.root_dir, args.data_dir, f'visdial_1.0_{split}_with_vispro_bert_tokenized.json')))

        dialogs = []
        for dialog_id, dialog in enumerate(data['data']['dialogs']):
            if 'pronoun_info' in dialog:
                dialogs.append(dialog)

        # save results
        filename = osp.join(args.root_dir, args.data_dir, f'visdial_1.0_{split}_with_vispro_only_bert_tokenized.json')
        data['data']['dialogs'] = dialogs
        with open(filename, 'w') as f:
            json.dump(data, f)
        print(f'{len(data["data"]["dialogs"])} dialogs copied. Result saved to {filename}')
