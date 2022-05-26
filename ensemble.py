import os
import os.path as osp
import numpy as np
import json
import glob
import argparse
import pyhocon
import glog as log
import re
import torch
from tqdm import tqdm

from utils.data_utils import load_pickle_lines
from utils.init_utils import set_log_file, copy_file_to_log 
from utils.visdial_metrics import SparseGTMetrics, NDCG, scores_to_ranks


parser = argparse.ArgumentParser(description='Ensemble for VisDial')
parser.add_argument('--exp', type=str,
                    help='experiment name from .conf')
parser.add_argument('--mode', type=str, choices=['eval', 'predict'],
                    help='eval or predict')
parser.add_argument('--ssh', action='store_true',
                    help='whether or not we are executing command via ssh. '
                         'If set to True, we will not log.info anything to screen and only redirect them to log file')


if __name__ == '__main__':
    args = parser.parse_args()

    # initialization
    config = pyhocon.ConfigFactory.parse_file(f"config/ensemble.conf")[args.exp]
    config["log_dir"] = os.path.join(config["log_dir"], args.exp)
    if not os.path.exists(config["log_dir"]):
        os.makedirs(config["log_dir"])

    # set logs
    log_file = os.path.join(config["log_dir"], f'{args.mode}.log')
    set_log_file(log_file, file_only=args.ssh)

    # print environment info
    log.info(f"Running experiment: {args.exp}")
    log.info(f"Results saved to {config['log_dir']}")
    log.info(pyhocon.HOCONConverter.convert(config, "hocon"))

    # load data
    if args.mode == 'eval':
        assert config['split'] == 'val'
        if config['processed']:
            dense_ann = json.load(open(config['visdial_val_dense_ann_processed']))
        else:
            dense_ann = json.load(open(config['visdial_val_dense_ann']))
        num_dialogs = len(dense_ann)
        imid2did = {d['image_id']: i for i, d in enumerate(dense_ann)}
        image_ids = imid2did.keys()
    if isinstance(config['processed'], list):
        assert len(config['models']) == len(config['processed'])
        processed = {model:pcd for model, pcd in zip(config['models'], config['processed'])}
    else:
        processed = {model: config['processed'] for model in config['models']}
    if config['split'] == 'test' and np.any(config['processed']):
        test_data = json.load(open(config['visdial_test_data']))['data']['dialogs']
        imid2rndid = {t['image_id']: len(t['dialog']) for t in test_data}
        del test_data

    # load predictions files
    visdial_outputs = dict()
    if args.mode == 'eval':
        metrics = {}
    for model in config['models']:
        pred_filename = osp.join(config['pred_dir'], model, 'visdial_prediction.pkl')
        pred_dict = {p['image_id']: p for p in load_pickle_lines(pred_filename)}
        log.info(f'Loading {len(pred_dict)} predictions from {pred_filename}')
        visdial_outputs[model] = pred_dict
        if args.mode == 'eval':
            assert len(visdial_outputs[model]) >= num_dialogs
            metric = json.load(open(osp.join(config['pred_dir'], model, "metrics_epoch_best.json")))
            metrics[model] = metric['val']

    if args.mode == 'eval':
        log.info(f'Calculating visdial metrics for {num_dialogs} dialogs')
        if not config['skip_mrr_eval']:
            sparse_metrics = SparseGTMetrics()
        ndcg = NDCG()
    else:
        image_ids = visdial_outputs[model].keys()
        predictions = []

    # for each dialog
    for image_id in tqdm(image_ids):
        scores = []
        round_id = None
        if args.mode == 'eval':
            dialog = dense_ann[imid2did[image_id]]
            gt_relevance = torch.Tensor(dialog['gt_relevance']).unsqueeze(0)
            gt_option_inds = None
        # load predictions
        for model in config['models']:
            pred = visdial_outputs[model][image_id]
            if args.mode == 'eval':
                gt_relevance_in_pred = torch.from_numpy(pred['gt_relevance']).unsqueeze(0)
                assert (gt_relevance_in_pred - gt_relevance).sum().item() < 1e-3, \
                    f'gt_relevance not match between gt and {model} {image_id}'
                gt_option_inds_in_pred = pred['gt_option_inds'][0]
                if gt_option_inds is None:
                    gt_option_inds = gt_option_inds_in_pred
                else:
                    assert gt_option_inds == gt_option_inds_in_pred
            if config['split'] == 'test' and processed[model]:
                # if predict on processed data, the first few rounds are deleted from some dialogs
                # so the original round ids can only be found in the original test data
                round_id_in_pred = imid2rndid[image_id]
            else:
                round_id_in_pred = pred['gt_relevance_round_id']
            if not isinstance(round_id_in_pred, int):
                round_id_in_pred = int(round_id_in_pred)
            if round_id is None:
                round_id = round_id_in_pred
            else:
                # make sure all models have the same round_id
                assert round_id == round_id_in_pred
            scores.append(torch.from_numpy(pred['nsp_probs']).unsqueeze(0))

        # ensemble scores
        scores = torch.cat(scores, 0) # [n_model, num_rounds, num_options]
        scores = torch.sum(scores, dim=0, keepdim=True) # [1, num_rounds, num_options]

        # output to score or evaluate
        if args.mode == 'eval':
            if not config['skip_mrr_eval']:
                assert scores.size(1) == 10
                gt_option_inds = torch.LongTensor([gt_option_inds]).unsqueeze(0)
                sparse_metrics.observe(scores, gt_option_inds)

            if scores.size(0) > 1:
                scores = scores[0, round_id - 1].unsqueeze(0) # [1, num_options]
            else:
                scores = scores.squeeze(0)

            ndcg.observe(scores, gt_relevance)

        else:
            if scores.size(0) > 1:
                scores = scores[round_id - 1].unsqueeze(0)
            ranks = scores_to_ranks(scores) # [eval_batch_size, num_rounds, num_options]
            ranks = ranks.squeeze(1)
            prediction = {
                "image_id": image_id,
                "round_id": round_id,
                "ranks": ranks[0].tolist()
            }
            predictions.append(prediction)

    # output results
    if args.mode == 'eval':
        metrics_esb = {}
        if not config['skip_mrr_eval']:
            metrics_esb.update(sparse_metrics.retrieve(reset=True))
        metrics_esb.update(ndcg.retrieve(reset=True))
        for model in config['models']:
            log.info(f'{model}: {metrics[model]}')
        log.info(f'ensemble: {metrics_esb}')
    else:
        filename = osp.join(config['log_dir'], f'{config["split"]}_ensemble_prediction.json')
        with open(filename, 'w') as f:
            json.dump(predictions, f)
        log.info(f'{len(predictions)} predictions saved to {filename}')
