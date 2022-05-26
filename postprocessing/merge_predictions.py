import os.path as osp
import re
import json
import numpy as np
import argparse
import pickle
import glob
from tqdm import tqdm

parser = argparse.ArgumentParser(description='turn multiple prediction files into one file')
parser.add_argument('--mode', type=str, choices=['phase1', 'phase2'], default='phase1',
                    help='phase1: merge multiple .jsonlines into one .json; phase2: merge multiple .npz into one .pkl')
parser.add_argument('--model', type=str, help='model name', default='MB-JC_predict')
parser.add_argument('--split', type=str, help='split name for phase2', choices=['val', 'test'], 
                    default='val')
parser.add_argument('--log_dir', type=str, help='log dir of the model', default='logs/vonly')
parser.add_argument('--output_dir', type=str, default='visdial_output_best', 
                    help='dir of .npz')

NUM_TEST_DIALOGS = 8000

if __name__ == "__main__":
    args = parser.parse_args()
    args.log_dir = osp.join(args.log_dir, args.model)

    if args.mode == 'phase1':
        prediction_files = glob.glob(osp.join(args.log_dir, 'visdial_prediction*.jsonlines'))
        all_visdial_predictions = []
        image_id_set = set()
        for prediciton_file in prediction_files:
            for line in open(prediciton_file):
                line = json.loads(line)
                if line['image_id'] not in image_id_set:
                    all_visdial_predictions.append(line)
                    image_id_set.add(line['image_id'])
        assert len(all_visdial_predictions) == 8000, f'Not all the test samples are covered in {args.log_dir}'
        visdial_file_name = osp.join(args.log_dir, 'visdial_prediction.json')
        with open(visdial_file_name, 'w') as f_visdial:
            json.dump(all_visdial_predictions, f_visdial, indent=2)
        print(f'Prediction for submisson save to {visdial_file_name}.')

    elif args.mode == 'phase2':
        dirname = osp.join(args.log_dir, args.output_dir, args.split, '*.npz')
        filenames = glob.glob(dirname)
        if len(filenames) == 0:
            dirname = dirname.replace('output_best', 'output')
            filenames = glob.glob(dirname)
        print(f'Merging {len(filenames)} .npz from {dirname}')

        output_filename = osp.join(args.log_dir, 'visdial_prediction.pkl')
        with open(output_filename, 'wb') as f:
            for filename in tqdm(filenames):
                image_id = int(re.search(r'(\d+)\.npz', filename).group(1))
                data_to_save = {'image_id': image_id}
                data = np.load(filename)
                for key in data.files:
                    data_to_save[key] = data[key]
                pickle.dump(data_to_save, f)

        print(f'Result saved to {output_filename}')

