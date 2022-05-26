import os
import os.path as osp
import json
from collections import deque, OrderedDict
import time
import re
import shutil
import glob
import pickle
import gc
import numpy as np
import glog as log
try:
    from apex import amp
except ModuleNotFoundError:
    print('apex not found')

import torch
import torch.utils.data as tud
import torch.nn.functional as F
import torch.distributed as dist

from utils.model_utils import TensorboardWriter
from utils.data_utils import load_pickle_lines
from utils.visdial_metrics import SparseGTMetrics, NDCG, scores_to_ranks
from utils.coref_metrics import CorefEvaluator, PrCorefEvaluator, gather_round_metrics


class Runner:
    def __init__(self, config):
        self.config = config
        if 'rank' in config:
            self.gpu_rank = config['rank']
        else:
            self.gpu_rank = 0

        self.epoch_idx = 0
        self.max_metric = 0.
        self.max_metric_epoch_idx = 0
        self.na_str = 'N/A'

        if self.config["max_ckpt_to_keep"] > 0:
            self.checkpoint_queue = deque([], maxlen=config["max_ckpt_to_keep"])
            self.metrics_queue = deque([], maxlen=config["max_ckpt_to_keep"])

        if (self.config['training'] or self.config['debugging']) and self.config['display']:
            self.writer = TensorboardWriter(config["log_dir"])

    def forward(self, batch, eval_visdial=False, eval_coref=False):
        return NotImplementedError

    def train(self, dataset, dataset_eval=None):
        if os.path.exists(self.config['log_dir']) or self.config['loads_ckpt'] or self.config['loads_best_ckpt']:
            self.load_ckpt()

        if self.config['use_trainval']:
            dataset.split = 'trainval'
        else:
            dataset.split = 'train'
        batch_size = self.config['batch_size']
        if self.config['parallel'] and self.config['dp_type'] != 'dp':
            sampler = tud.distributed.DistributedSampler(
                            dataset,
                            num_replicas=self.config['num_gpus'],
                            rank=self.gpu_rank
                        )
        else:
            sampler = None

        data_loader = tud.DataLoader(
                    dataset=dataset,
                    batch_size=batch_size,
                    shuffle=self.config['training'] and not self.config['parallel'],
                    # pin_memory=True,
                    collate_fn=dataset.collate_fn,
                    num_workers=self.config['num_workers'],
                    sampler=sampler
                )

        start_epoch_idx = self.epoch_idx
        num_iter_epoch = self.config['num_iter_per_epoch']
        if self.config['display']:
            log.info(f'{num_iter_epoch} iter per epoch.')

        # eval before training
        eval_dense_at_first = self.config['train_on_dense'] and self.config['skip_mrr_eval'] and start_epoch_idx == 0
        # eval before training under 2 circumstances:
        # for dense finetuning, eval ndcg before the first epoch
        # for mrr training, continue training and the last epoch is not evaluated
        if not self.config['skip_eval'] and (eval_dense_at_first or (self.config['eval_at_start'] and len(self.metrics_queue) == 0 and start_epoch_idx > 0)):
            if eval_dense_at_first:
                iter_now = 0
            # elif start_epoch_idx > 0:
            else:
                iter_now = max(num_iter_epoch * start_epoch_idx, 0)

            if dataset_eval is None:
                dataset.split = 'val'
                dataset_to_eval = dataset
            else:
                dataset_to_eval = dataset_eval
            metrics_results = {}
            metrics_to_maximize, metrics_results['val'] = self.evaluate(dataset_to_eval, iter_now)
            if eval_dense_at_first:
                self.max_metric = metrics_to_maximize
                self.max_metric_epoch_idx = -1
            else:
                if self.config['display']:
                    self.save_eval_results('val', start_epoch_idx - 1, metrics_results)
                    if metrics_to_maximize > self.max_metric:
                        self.max_metric = metrics_to_maximize
                        self.max_metric_epoch_idx = start_epoch_idx - 1
                        self.copy_best_results('val', start_epoch_idx - 1)
                        self.copy_best_predictions('val')
            if dataset_eval is None:
                if self.config['use_trainval']:
                    dataset.split = 'trainval'
                else:
                    dataset.split = 'train'

        for epoch_idx in range(start_epoch_idx, self.config['num_epochs']):
            if self.config['parallel'] and self.config['dp_type'] != 'dp':
                sampler.set_epoch(epoch_idx)
                # if self.config['dp_type'] == 'apex':
                #     gc.collect()
                #     torch.cuda.empty_cache()

            self.epoch_idx = epoch_idx

            if self.config['display']:
                log.info(f'starting epoch {epoch_idx}')
                log.info('training')

            self.model.train()

            num_batch = 0
            next_logging_pct = .1
            next_evaluating_pct = self.config["next_evaluating_pct"] + .1
            start_time = time.time()
            self.optimizer.zero_grad()

            for batch in data_loader:
                num_batch += 1
                pct = num_batch / num_iter_epoch * 100
                iter_now = num_iter_epoch * epoch_idx + num_batch

                output = self.forward(batch)
                losses = output['losses']

                # optimizer step
                losses['tot_loss'] /= self.config['batch_multiply']
                # debug
                if self.config['debugging']:
                    log.info('try backward')
                if self.config['dp_type'] == 'apex':
                    with amp.scale_loss(losses['tot_loss'], self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    losses['tot_loss'].backward()
                if self.config['debugging']:
                    log.info('backward done')
                self.scheduler.step()

                if iter_now % self.config['batch_multiply'] == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # display and eval
                if pct >= next_logging_pct:
                    if self.config['display']:
                        loss_to_print = ''
                        for key in losses:
                            if losses[key] is not None and isinstance(losses[key], torch.Tensor):
                                loss_to_print += f'{key}: {losses[key].item():.4f} '
                                self.writer.add_scalar(f'Train/{key}', losses[key].item(), iter_now)
                        log.info(
                            f'{int(pct)}%, time: {time.time() - start_time:.2f} {loss_to_print}'
                        )

                        # bert_lr, task_lr = self.optimizer.learning_rate()
                        # self.writer.add_scalar('Train/lr_bert', bert_lr, iter_now)
                        # self.writer.add_scalar('Train/lr_task', task_lr, iter_now)
                        lr = self.scheduler.get_lr()[0]
                        self.writer.add_scalar('Train/lr', lr, iter_now)

                    next_logging_pct += self.config["next_logging_pct"]

                    if self.config['debugging']:
                        break

                if pct >= next_evaluating_pct and not self.config['skip_eval']:
                    if self.config['use_coref']:
                        if dataset_eval is None:
                            dataset.split = 'val'
                            dataset_to_eval = dataset
                        else:
                            dataset_to_eval = dataset_eval
                        self.evaluate(dataset_to_eval, iter_now, eval_visdial=False)
                        if dataset_eval is None:
                            if self.config['use_trainval']:
                                dataset.split = 'trainval'
                            else:
                                dataset.split = 'train'

                    next_evaluating_pct += self.config["next_evaluating_pct"]

                del losses
                # debug
                # torch.cuda.empty_cache()

            if self.config['display']:
                log.info(
                    f'100%,\ttime:\t{time.time() - start_time:.2f}'
                )
                ckpt_path = self.save_ckpt()

            if not self.config['skip_eval']:

                iter_now = num_iter_epoch * (epoch_idx + 1)

                if dataset_eval is None:
                    dataset.split = 'val'
                    dataset_to_eval = dataset
                else:
                    dataset_to_eval = dataset_eval
                metrics_results = {}
                metrics_to_maximize, metrics_results['val'] = self.evaluate(dataset_to_eval, iter_now)
                if dataset_eval is None:
                    if self.config['use_trainval']:
                        dataset.split = 'trainval'
                    else:
                        dataset.split = 'train'
                if self.config['display']:
                    self.save_eval_results('val', epoch_idx, metrics_results)

                if self.config['display']:

                    if metrics_to_maximize > self.max_metric:
                        self.max_metric = metrics_to_maximize
                        self.max_metric_epoch_idx = epoch_idx
                        self.copy_best_results('val', epoch_idx)
                        self.copy_best_predictions('val')

                    elif not self.config['parallel'] and epoch_idx - self.max_metric_epoch_idx > self.config["early_stop_epoch"]:
                        self.writer.add_scalar('Val/metric_best', self.max_metric, iter_now)
                        log.info('Early stop.')
                        break

                    self.writer.add_scalar('Val/metric_best', self.max_metric, iter_now)

            if self.config['parallel']:
                if self.config['dp_type'] == 'dp':
                    gc.collect()
                    torch.cuda.empty_cache()
                else:
                    dist.barrier()

            if self.config['stop_epochs'] >= 0 and epoch_idx + 1 >= self.config['stop_epochs']:
                if self.config['display']:
                    log.info('Stop for reaching stop_epochs.')
                break


    def evaluate(self, dataset, training_iter=None, eval_visdial=True):
        if self.config['model_type'] == 'conly':
            eval_visdial = False

        # create files to save output
        if self.config['predicting']:
            visdial_file_name = None
            coref_file_name = None
            if self.config['save_score']:
                if self.config['model_type'] in ['vonly', 'joint']:
                    visdial_file_name = osp.join(self.config['log_dir'], f'visdial_prediction.pkl')
                    if osp.exists(visdial_file_name):
                        dialogs_predicted = load_pickle_lines(visdial_file_name)
                        dialogs_predicted = [d['image_id'] for d in dialogs_predicted]
                    else:
                        dialogs_predicted = []
                    f_visdial = open(visdial_file_name, 'ab')
                if self.config['model_type'] in ['conly', 'joint']:
                    raise NotImplementedError()
            else:
                if self.config['model_type'] in ['vonly', 'joint']:
                    visdial_file_name = osp.join(self.config['log_dir'], f'visdial_prediction.jsonlines')
                    if self.config['parallel'] and self.config['dp_type'] != 'dp':
                        visdial_file_name = visdial_file_name.replace('.jsonlines', f'_{self.config["rank"]}of{self.config["num_gpus"]}.jsonlines')
                    if osp.exists(visdial_file_name):
                        dialogs_predicted_visdial = [json.loads(line)['image_id'] for line in open(visdial_file_name)]
                        f_visdial = open(visdial_file_name, 'a')
                    else:
                        dialogs_predicted_visdial = []
                        f_visdial = open(visdial_file_name, 'w')
                if self.config['model_type'] in ['conly', 'joint']:
                    coref_file_name = osp.join(self.config['log_dir'], f'{dataset.split}_coref_prediction.jsonlines')
                    if self.config['parallel'] and self.config['dp_type'] != 'dp':
                        coref_file_name = coref_file_name.replace('.jsonlines', f'_{self.config["rank"]}of{self.config["num_gpus"]}.jsonlines')
                    if osp.exists(coref_file_name):
                        dialogs_predicted_coref = [json.loads(line)['image_id'] for line in open(coref_file_name)]
                        f_coref = open(coref_file_name, 'a')
                    else:
                        dialogs_predicted_coref = []
                        f_coref = open(coref_file_name, 'w')
                if self.config['model_type'] == 'joint':
                    dialogs_predicted = [d for d in dialogs_predicted_visdial if d in dialogs_predicted_coref]
                elif self.config['model_type'] == 'vonly':
                    dialogs_predicted = dialogs_predicted_visdial
                elif self.config['model_type'] == 'conly':
                    dialogs_predicted = dialogs_predicted_coref
            if len(dialogs_predicted) > 0:
                log.info(f'Found {len(dialogs_predicted)} predicted results.')
            if self.config['display']:
                if visdial_file_name is not None:
                    log.info(f'VisDial predictions saved to {visdial_file_name}')
                if coref_file_name is not None:
                    log.info(f'VisPro predictions saved to {coref_file_name}')

        elif self.config['display']:
            if self.config['continue_evaluation']:
                if self.config['model_type'] in ['vonly', 'joint']:
                    predicted_files = os.listdir(osp.join(self.config['visdial_output_dir'], dataset.split))
                    dialogs_predicted = [int(re.match(r'(\d+).npz', p).group(1)) for p in predicted_files]
                if self.config['model_type'] in ['conly', 'joint']:
                    raise NotImplementedError
            else:
                if self.config['model_type'] in ['vonly', 'joint']:
                    if osp.exists(osp.join(self.config['visdial_output_dir'], dataset.split)):
                        shutil.rmtree(osp.join(self.config['visdial_output_dir'], dataset.split))
                    os.makedirs(osp.join(self.config['visdial_output_dir'], dataset.split))
                if self.config['model_type'] in ['conly', 'joint']:
                    if osp.exists(osp.join(self.config['coref_output_dir'], dataset.split)):
                        shutil.rmtree(osp.join(self.config['coref_output_dir'], dataset.split))
                    os.makedirs(osp.join(self.config['coref_output_dir'], dataset.split))
                dialogs_predicted = []
            log.info(f'Found {len(dialogs_predicted)} predicted results.')

        if self.config['parallel'] and self.config['dp_type'] != 'dp':
            sampler = tud.distributed.DistributedSampler(
                            dataset,
                            num_replicas=self.config['num_gpus'],
                            rank=self.gpu_rank
                        )

            dist.barrier()
        else:
            sampler = None
        data_loader = tud.DataLoader(
                    dataset=dataset,
                    batch_size=self.config['eval_batch_size'],
                    shuffle=False,
                    # pin_memory=True,
                    collate_fn=dataset.collate_fn,
                    num_workers=self.config['num_workers'],
                    sampler=sampler
                )

        self.model.eval()

        with torch.no_grad():
            if self.config['display']:
                log.info(f'Evaluating {len(dataset)} samples')

            avg_coref_loss = 0.
            next_logging_pct = self.config["next_logging_pct"] + .1
            if self.config['parallel'] and self.config['dp_type'] == 'dp':
                num_batch_tot = int(np.ceil(len(dataset) / self.config['eval_batch_size']))
            else:
                num_batch_tot = int(np.ceil(len(dataset) / (self.config['eval_batch_size'] * self.config['num_gpus'])))
            num_batch = 0
            if dataset.split == 'val':
                num_options = self.config["num_options"]
                if self.config['skip_mrr_eval']:
                    num_rounds = 1
                else:
                    num_rounds = 10
            elif dataset.split == 'test':
                num_options = 100
                num_rounds = 1
            start_time = time.time()

            for batch in data_loader:
                num_batch += 1
                # skip dialogs that have been predicted
                if self.config['predicting']:
                    image_ids = batch['image_id'].tolist()
                    skip_batch = True
                    for image_id in image_ids:
                        if image_id not in dialogs_predicted:
                            skip_batch = False
                    if skip_batch:
                        continue

                output = self.forward(batch, eval_coref=True, eval_visdial=eval_visdial)

                # visdial evaluation
                if eval_visdial:
                    img_ids = batch['image_id'].tolist()
                    batch_size = len(img_ids)
                    gt_relevance_round_id = batch['round_id'].tolist()

                    nsp_scores = output['nsp_scores'] # [batch_size * num_rounds * num_options, 2]
                    nsp_probs = F.softmax(nsp_scores, dim=1)
                    assert nsp_probs.shape[-1] == 2
                    nsp_probs = nsp_probs[:, 0] # num_dim=2, 0 for postive, 1 for negative
                    nsp_probs = nsp_probs.view(batch_size, num_rounds, num_options)

                    # could be predicting or evaluating
                    if dataset.split == 'val':
                        gt_option_inds = batch['gt_option_inds'] # [batch_size, num_rounds]
                        gt_relevance = batch['gt_relevance'] # [batch_size, num_options]

                        for b in range(batch_size):
                            filename = osp.join(self.config['visdial_output_dir'], dataset.split, f'{img_ids[b]}.npz')
                            if not osp.exists(filename):
                                np.savez(filename, 
                                         nsp_probs=nsp_probs[b].cpu().numpy(), 
                                         gt_option_inds=gt_option_inds[b].cpu().numpy(),
                                         gt_relevance=gt_relevance[b].cpu().numpy(),
                                         gt_relevance_round_id=gt_relevance_round_id[b])

                    # must be predicting
                    if dataset.split == 'test':
                        if self.config['save_score']:
                            for b in range(batch_size):
                                prediction = {
                                    "image_id": img_ids[b],
                                    "nsp_probs": nsp_probs[b].cpu().numpy(),
                                    "gt_relevance_round_id": gt_relevance_round_id[b]
                                }
                                pickle.dump(prediction, f_visdial)
                        else:
                            ranks = scores_to_ranks(nsp_probs) # [eval_batch_size, num_rounds, num_options]
                            ranks = ranks.squeeze(1)
                            for b in range(batch_size):
                                prediction = {
                                    "image_id": img_ids[b],
                                    "round_id": gt_relevance_round_id[b],
                                    "ranks": ranks[b].tolist()
                                }
                                f_visdial.write(json.dumps(prediction) + '\n')

                # coref evaluation
                if self.config['use_coref']:
                    num_dialog = len(batch['dialog_info'])
                    coref_loss = output['losses']['coref_loss']
                    coref_predictions = output['coref_predictions']
                    if isinstance(coref_loss, torch.Tensor):
                        avg_coref_loss += coref_loss.item() * num_dialog

                    if self.config['predicting']:
                        _, image_id, _, _ = batch['dialog_info'][0]
                        if not self.config['predict_each_round']:
                            predicted_clusters = coref_predictions[0]
                        else:
                            predicted_clusters = coref_predictions
                        res = {'image_id': image_id, 'predicted_clusters': predicted_clusters}
                        f_coref.write(json.dumps(res) + '\n')
                    else:
                        for dialog_id in range(num_dialog):
                            if self.config['train_each_round']:
                                predicted_clusters = coref_predictions
                                coref_info = []
                                for dialog_info_rnd, pred_clusters_rnd in zip(batch['dialog_info'][dialog_id], predicted_clusters):
                                    gold_clusters, image_id, pronoun_info, sentences = dialog_info_rnd
                                    coref_info.append({
                                                       'gold_clusters': gold_clusters,
                                                       'predicted_clusters': pred_clusters_rnd,
                                                       'pronoun_info': pronoun_info,
                                                       'sentences': sentences,
                                                      })
                            else:
                                predicted_clusters = coref_predictions[dialog_id]
                                gold_clusters, image_id, pronoun_info, sentences = batch['dialog_info'][dialog_id]
                                if pronoun_info is None:
                                    continue

                                coref_info = {
                                    'gold_clusters': gold_clusters,
                                    'predicted_clusters': predicted_clusters,
                                    'pronoun_info': pronoun_info,
                                    'sentences': sentences,
                                }
                            filename = osp.join(self.config['coref_output_dir'], dataset.split, f'{image_id}.json')
                            if not osp.exists(filename):
                                with open(filename, 'w') as f:
                                    json.dump(coref_info, f)

                # debug
                if self.config['debugging']:
                    break

                pct = num_batch / num_batch_tot * 100
                if pct >= next_logging_pct:
                    if self.config['display']:
                        log.info(
                            f'{int(pct)}%,\ttime:\t{time.time() - start_time:.2f}'
                        )
                    next_logging_pct += self.config["next_logging_pct"]
                    # debug
                    if self.config['debugging']:
                        break

        if self.config['display']:
            pct = num_batch / num_batch_tot * 100
            log.info(
                f'{int(pct)}%,\ttime:\t{time.time() - start_time:.2f}'
            )

        if not self.config['validating']:
            self.model.train()

        if self.config['parallel'] and self.config['dp_type'] != 'dp':
            dist.barrier()

        if self.config['predicting']:
            if self.config['model_type'] in ['vonly', 'joint']:
                f_visdial.close()
                if not self.config['save_score']:
                    all_visdial_predictions = [json.loads(line) for line in open(visdial_file_name)]
                    if self.config['predict_split'] == 'test' and len(all_visdial_predictions) == self.config['num_test_dialogs']:
                        visdial_file_name = visdial_file_name.replace('jsonlines', 'json')
                        with open(visdial_file_name, 'w') as f_visdial:
                            json.dump(all_visdial_predictions, f_visdial)
                        log.info(f'Prediction for submisson save to {visdial_file_name}.')
            if self.config['model_type'] in ['conly', 'joint'] and not self.config['save_score']:
                f_coref.close()
            return None, None

        if self.config['display']:
            if dataset.split == 'val' and eval_visdial:
                if not self.config['skip_mrr_eval']:
                    sparse_metrics = SparseGTMetrics()
                ndcg = NDCG()
            if self.config['use_coref'] and not self.config['predicting']:
                if self.config['train_each_round']:
                    coref_evaluator = [CorefEvaluator() for _ in range(10)]
                    pr_coref_evaluator = [PrCorefEvaluator() for _ in range(10)]
                else:
                    coref_evaluator = CorefEvaluator()
                    pr_coref_evaluator = PrCorefEvaluator()

            if dataset.split == 'val' and eval_visdial:
                visdial_output_filenames = glob.glob(osp.join(self.config['visdial_output_dir'], dataset.split, '*.npz'))
                log.info(f'Calculating visdial metrics for {len(visdial_output_filenames)} dialogs')
                for visdial_output_filename in visdial_output_filenames:
                    output = np.load(visdial_output_filename)
                    nsp_probs = torch.from_numpy(output['nsp_probs']).unsqueeze(0)
                    gt_relevance = torch.from_numpy(output['gt_relevance']).unsqueeze(0)
                    if not self.config['skip_mrr_eval']:
                        gt_option_inds = torch.from_numpy(output['gt_option_inds']).unsqueeze(0)
                        gt_relevance_round_id = output['gt_relevance_round_id']
                        sparse_metrics.observe(nsp_probs, gt_option_inds)
                        nsp_probs_dense = nsp_probs[0, gt_relevance_round_id - 1, :].unsqueeze(0)
                    else:
                        nsp_probs_dense = nsp_probs.squeeze(0) # [1, 100]
                    ndcg.observe(nsp_probs_dense, gt_relevance)

            if self.config['use_coref'] and not self.config['predicting']:
                coref_output_filenames = glob.glob(osp.join(self.config['coref_output_dir'], dataset.split, '*.json'))
                log.info(f'Calculating coref metrics for {len(coref_output_filenames)} dialogs')
                for coref_output_filename in coref_output_filenames:
                    output = json.load(open(coref_output_filename))

                    if self.config['train_each_round']:
                        for rnd, output_rnd in enumerate(output):
                            self.update_coref_evaluator(output_rnd, 
                                                        coref_evaluator[rnd], 
                                                        pr_coref_evaluator[rnd])
                    else:
                        self.update_coref_evaluator(output, coref_evaluator, pr_coref_evaluator)

            # visdial eval output
            visdial_metrics = {}
            if dataset.split == 'val' and eval_visdial:
                if not self.config['skip_mrr_eval']:
                    visdial_metrics.update(sparse_metrics.retrieve(reset=True))
                visdial_metrics.update(ndcg.retrieve(reset=True))

                if self.config['display']:
                    to_print = ''
                    for metric_name, metric_value in visdial_metrics.items():
                        if 'round' not in metric_name:
                            to_print += f"\n{metric_name}: {metric_value}"
                            if training_iter is not None:
                                self.writer.add_scalar('Val/' + metric_name.replace('@', '_'), metric_value, training_iter)
                    log.info(to_print)

            # coref eval output
            if self.config['use_coref']:
                avg_coref_loss /= num_batch_tot
                if self.config['train_each_round']:
                    (coref_precision_rnds, coref_recall_rnds, coref_f1_rnds), \
                            (coref_precision, coref_recall, coref_f1) = \
                            gather_round_metrics(coref_evaluator)
                    (prp_precision_rnds, prp_recall_rnds, prp_f1_rnds), \
                            (prp_precision, prp_recall, prp_f1) = \
                            gather_round_metrics(pr_coref_evaluator)
                else:
                    coref_precision, coref_recall, coref_f1 = coref_evaluator.get_prf()
                    prp_precision, prp_recall, prp_f1 = pr_coref_evaluator.get_prf()

                if self.config['display']:
                    log.info('\n'
                        f'avg_valid_time:\t{time.time() - start_time:.2f}\n'
                        f'avg loss:\t{avg_coref_loss:.4f}\n'
                        f'Coref average precision:\t{coref_precision:.4f}\n'
                        f'Coref average recall:\t{coref_recall:.4f}\n'
                        f'Coref average f1:\t{coref_f1:.4f}\n'
                    )

                    if training_iter is not None:
                        self.writer.add_scalar('Val/coref_loss', avg_coref_loss, training_iter)
                        self.writer.add_scalar('Val/coref_precision', coref_precision, training_iter)
                        self.writer.add_scalar('Val/coref_recall', coref_recall, training_iter)
                        self.writer.add_scalar('Val/coref_f1', coref_f1, training_iter)


                    log.info('\n'
                        f'Pronoun_Coref_average_precision:\t{prp_precision:.4f}\n'
                        f'Pronoun_Coref_average_recall:\t{prp_recall:.4f}\n'
                        f'Pronoun_Coref_average_f1:\t{prp_f1:.4f}\n'
                    )

                    if training_iter is not None:
                        self.writer.add_scalar('Val/pronoun_coref_precision', prp_precision, training_iter)
                        self.writer.add_scalar('Val/pronoun_coref_recall', prp_recall, training_iter)
                        self.writer.add_scalar('Val/pronoun_coref_f1', prp_f1, training_iter)

                    results = {'coref_p': coref_precision, 'coref_r': coref_recall, 'coref_f1': coref_f1,
                           'prp_p': prp_precision, 'prp_r': prp_recall, 'prp_f1': prp_f1}

                    if self.config['train_each_round']:

                        log.info('\n'
                            f'Coref average precision rounds:\t{coref_precision_rnds:.4f}\n'
                            f'Coref average recall rounds:\t{coref_recall_rnds:.4f}\n'
                            f'Coref average f1 rounds:\t{coref_f1_rnds:.4f}\n'
                        )

                        if training_iter is not None:
                            self.writer.add_scalar('Val/round_coref_precision', coref_precision_rnds, training_iter)
                            self.writer.add_scalar('Val/round_coref_recall', coref_recall_rnds, training_iter)
                            self.writer.add_scalar('Val/round_coref_f1', coref_f1_rnds, training_iter)


                        log.info('\n'
                            f'Pronoun_Coref_average_precision rounds:\t{prp_precision_rnds:.4f}\n'
                            f'Pronoun_Coref_average_recall rounds:\t{prp_recall_rnds:.4f}\n'
                            f'Pronoun_Coref_average_f1 rounds:\t{prp_f1_rnds:.4f}\n'
                        )

                        if training_iter is not None:
                            self.writer.add_scalar('Val/round_pronoun_coref_precision', prp_precision_rnds, training_iter)
                            self.writer.add_scalar('Val/round_pronoun_coref_recall', prp_recall_rnds, training_iter)
                            self.writer.add_scalar('Val/round_pronoun_coref_f1', prp_f1_rnds, training_iter)

                        results.update({'coref_p_rnds': coref_precision_rnds, 'coref_r_rnds': coref_recall_rnds, 'coref_f1_rnds': coref_f1_rnds,
                           'prp_p_rnds': prp_precision_rnds, 'prp_r_rnds': prp_recall_rnds, 'prp_f1_rnds': prp_f1_rnds})
            else:
                results = {}

            results.update(visdial_metrics)
            if self.config['metrics_to_maximize'] in results:
                metrics_to_maximize = results[self.config['metrics_to_maximize']]
            else:
                metrics_to_maximize = None

            return metrics_to_maximize, results
        else:
            return None, None

    def update_coref_evaluator(self, output, coref_evaluator, pr_coref_evaluator):
        gold_clusters = output['gold_clusters']
        predicted_clusters = output['predicted_clusters']
        pronoun_info = output['pronoun_info']
        sentences = output['sentences']

        gold_clusters = [
            tuple(tuple(span) for span in cluster)
            for cluster in gold_clusters
        ]
        span_to_gold_cluster = {
            span: cluster
            for cluster in gold_clusters
            for span in cluster
        }
        predicted_clusters = [
            tuple(tuple(span) for span in cluster)
            for cluster in predicted_clusters
        ]
        span_to_predicted_cluster = {
            span: cluster
            for cluster in predicted_clusters
            for span in cluster
        }
        coref_evaluator.update(
            predicted=predicted_clusters,
            gold=gold_clusters,
            mention_to_predicted=span_to_predicted_cluster,
            mention_to_gold=span_to_gold_cluster
        )
        pr_coref_evaluator.update(predicted_clusters, pronoun_info, sentences)

    def save_eval_results(self, split, epoch_idx, metrics_results):

        metrics_filename = osp.join(self.config['log_dir'], f'metrics_epoch_{epoch_idx}.json')
        with open(metrics_filename, 'w') as f:
            json.dump(metrics_results, f)
        log.info(f'Results of metrics saved to {metrics_filename}')

        if self.config["max_ckpt_to_keep"] > 0:
            if len(self.metrics_queue) == self.metrics_queue.maxlen:
                todel = self.metrics_queue.popleft()
                os.remove(todel)
            self.metrics_queue.append(metrics_filename)

        if epoch_idx == 'best':
            self.copy_best_predictions(split)

    def copy_best_results(self, split, epoch_idx):
        to_print = 'Copy '

        if not self.config['skip_saving_ckpt']:
            ckpt_path = osp.join(self.config['log_dir'], f'epoch_{epoch_idx}.ckpt')
            best_ckpt_path = ckpt_path.replace(f'{epoch_idx}.ckpt', 'best.ckpt')
            shutil.copyfile(ckpt_path, best_ckpt_path)
            to_print += best_ckpt_path + ' '

        metrics_filename = osp.join(self.config['log_dir'], f'metrics_epoch_{epoch_idx}.json')
        best_metric_filename = metrics_filename.replace(f'{epoch_idx}.json', 'best.json')
        shutil.copyfile(metrics_filename, best_metric_filename)
        to_print += best_metric_filename + ' '

        log.info(to_print)

    def copy_best_predictions(self, split):
        to_print = 'Copy '

        if self.config['use_coref']:
            coref_output_dir = osp.join(self.config['coref_output_dir'], split)
            if osp.exists(coref_output_dir):
                dir_best = coref_output_dir.replace('output', 'output_best')
                if osp.exists(dir_best):
                    shutil.rmtree(dir_best)
                shutil.copytree(coref_output_dir, dir_best)
                to_print += dir_best + ' '

        visdial_output_dir = osp.join(self.config['visdial_output_dir'], split)
        if osp.exists(visdial_output_dir):
            dir_best = visdial_output_dir.replace('output', 'output_best')
            if osp.exists(dir_best):
                shutil.rmtree(dir_best)
            shutil.copytree(visdial_output_dir, dir_best)
            to_print += dir_best + ' '

        log.info(to_print)

    def get_ckpt(self):
        ckpt = {
            'epoch_idx': self.epoch_idx,
            'max_metric': self.max_metric,
            'seed': self.config['random_seed'],
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        if self.config['parallel']:
            ckpt['model_state_dict'] = self.model.module.state_dict()
        else:
            ckpt['model_state_dict'] = self.model.state_dict()
        if self.config['dp_type'] == 'apex':
            ckpt['amp'] = amp.state_dict()
        return ckpt

    def set_ckpt(self, ckpt_dict):
        if not self.config['restarts']:
            self.epoch_idx = ckpt_dict['epoch_idx'] + 1

        if not self.config['resets_max_metric']:
            self.max_metric = ckpt_dict['max_metric']

        if self.config['parallel']:
            model = self.model.module
        else:
            model = self.model
        
        model_state_dict = model.state_dict()
        former_dict = {k: v for k, v in ckpt_dict['model_state_dict'].items() if k in model_state_dict}

        if self.config['display']:
            log.info("number of keys transferred: %d" % len(former_dict))
        assert len(former_dict.keys()) > 0

        model_state_dict.update(former_dict)

        model.load_state_dict(model_state_dict)
        if self.config['display']:
            log.info('loaded model')
        del model_state_dict, former_dict

        if not self.config['validating'] and not (self.config['uses_new_optimizer'] or self.config['sets_new_lr']):
            self.optimizer.load_state_dict(ckpt_dict['optimizer'])
            if self.config['display']:
                log.info('loaded optimizer')
            if 'scheduler' in ckpt_dict:
                self.scheduler.last_epcoh = ckpt_dict['epoch_idx'] * self.config['num_iter_per_epoch']
                self.scheduler.load_state_dict(ckpt_dict['scheduler'])

        if 'amp' in ckpt_dict and self.config['dp_type'] == 'apex':
            amp.load_state_dict(ckpt_dict['amp'])

        del ckpt_dict

        torch.cuda.empty_cache()

    # ckpt = property(get_ckpt, set_ckpt)

    def save_ckpt(self):
        ckpt_path = f'{self.config["log_dir"]}/epoch_{self.epoch_idx}.ckpt'
        log.info(f'saving checkpoint {ckpt_path}')
        ckpt = self.get_ckpt()
        if self.config['skip_saving_ckpt']:
            return ckpt_path
        torch_version = float(torch.__version__[:3])
        if torch_version - 1.4 > 1e-3:
            torch.save(ckpt, f=ckpt_path, _use_new_zipfile_serialization=False)
        else:
            torch.save(ckpt, f=ckpt_path)
        del ckpt

        if not (self.config['parallel'] and self.config['dp_type'] in ['ddp', 'apex']):
            torch.cuda.empty_cache()

        if self.config["max_ckpt_to_keep"] > 0:
            if len(self.checkpoint_queue) == self.checkpoint_queue.maxlen:
                todel = self.checkpoint_queue.popleft()
                os.remove(todel)
            self.checkpoint_queue.append(ckpt_path)

        return ckpt_path

    def save_ckpt_best(self):
        ckpt_path = f'{self.config["log_dir"]}/epoch_best.ckpt'
        log.info(f'saving checkpoint {ckpt_path}')
        ckpt = self.get_ckpt()
        torch.save(ckpt, f=ckpt_path)
        del ckpt
        return ckpt_path

    def load_ckpt_best(self):
        ckpt_path = f'{self.config["log_dir"]}/epoch_best.ckpt'
        if not osp.exists(ckpt_path):
            ckpt_paths = [path for path in os.listdir(f'{self.config["log_dir"]}/') if path.endswith('.ckpt') and 'best' not in path]
            if len(ckpt_paths) == 0:
                if self.config['display']:
                    log.info(f'No .ckpt found in {self.config["log_dir"]}')
                return
            sort_func = lambda x:int(re.search(r"(\d+)", x).groups()[0])
            ckpt_path = f'{self.config["log_dir"]}/{sorted(ckpt_paths, key=sort_func, reverse=True)[0]}'
        if self.config['display']:
            log.info(f'loading checkpoint {ckpt_path}')
        map_location = {'cuda:0': f'cuda:{self.gpu_rank}'}
        self.set_ckpt(torch.load(ckpt_path, map_location=map_location))

    def load_ckpt(self, ckpt_path=None):
        if not ckpt_path:
            if self.config['validating'] or self.config['loads_best_ckpt']:
                ckpt_path = f'{self.config["log_dir"]}/epoch_best.ckpt'
            else:
                ckpt_paths = [path for path in os.listdir(f'{self.config["log_dir"]}/') if path.endswith('.ckpt') and 'best' not in path]
                if len(ckpt_paths) == 0:
                    if self.config['display']:
                        log.info(f'No .ckpt found in {self.config["log_dir"]}')
                    return
                sort_func = lambda x:int(re.search(r"(\d+)", x).groups()[0])
                ckpt_path = f'{self.config["log_dir"]}/{sorted(ckpt_paths, key=sort_func, reverse=True)[0]}'

        if self.config['display']:
            log.info(f'loading checkpoint {ckpt_path}')
            epoch_name = osp.split(ckpt_path)[1].split('.')[0]
            if re.search(r"(\d+)", epoch_name):
                self.checkpoint_queue.append(ckpt_path)
                metrics_filename = osp.join(self.config['log_dir'], f'metrics_{epoch_name}.json')
                if osp.exists(metrics_filename):
                    self.metrics_queue.append(metrics_filename)

        map_location = {'cuda:0': f'cuda:{self.gpu_rank}'}
        self.set_ckpt(torch.load(ckpt_path, map_location=map_location))

    def match_model_key(self, pretrained_dict, model_dict):
        matched_dict = dict()
        for key in pretrained_dict:
            if key in model_dict:
                matched_key = key
            elif key.startswith('encoder.') and key[8:] in model_dict:
                matched_key = key[8:]
            elif key.startswith('module.') and key[7:] in model_dict:
                matched_key = key[7:]
            elif 'encoder.' + key in model_dict:
                matched_key = 'encoder.' + key
            elif 'module.' + key in model_dict:
                matched_key = 'module.' + key
            else:
                continue
            matched_dict[matched_key] = pretrained_dict[key]
        return matched_dict

    def load_pretrained_vilbert(self):

        if self.config['training'] or self.config['debugging']:
            # for coref bert
            if self.config['model_type'] == 'conly' \
                and self.config['use_embedding'] == 'bert' \
                and not self.config['loads_start_path']:
                return

            ckpt_paths = [path for path in os.listdir(f'{self.config["log_dir"]}/') if path.endswith('.ckpt') and 'best' not in path]
            if len(ckpt_paths) > 0:
                if self.config['display']:
                    log.info('Continue training')
                return

        if self.config['display']:
            log.info(f'Loading pretrained VilBERT from {self.config["start_path"]}')
        map_location = {'cuda:0': f'cuda:{self.gpu_rank}'}
        pretrained_dict = torch.load(self.config['start_path'], map_location=map_location)
        if 'model_state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['model_state_dict']
        if self.config['parallel']:
            model = self.model.module
        else:
            model = self.model
        model_dict = model.state_dict()

        matched_dict = self.match_model_key(pretrained_dict, model_dict)

        if self.config['display']:
            log.info("number of keys transferred: %d" % len(matched_dict))
        assert len(matched_dict.keys()) > 0
        model_dict.update(matched_dict)
        model.load_state_dict(model_dict)

        del pretrained_dict, model_dict, matched_dict
        if not self.config['parallel'] or self.config['dp_type'] == 'dp':
            torch.cuda.empty_cache()

        if self.config['display']:
            log.info(f'Pretrained VilBERT loaded')
