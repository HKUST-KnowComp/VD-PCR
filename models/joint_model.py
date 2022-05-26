import math
from functools import cmp_to_key, partial
import time
import sys
from collections import OrderedDict
import glog as log

import torch
from torch import nn
import torch.nn.functional as F
from pytorch_transformers import BertModel

sys.path.append('../')

from utils.model_utils import init_params
from utils.data_utils import build_len_mask_batch, sequence_mask
from utils.optim_utils import init_optim
from utils.modules import Squeezer
from models.coref_model import CorefModel
from models.visual_dialog_encoder import VisualDialogEncoderWithEmbedding
from models.runner import Runner


class JointModel(CorefModel):
    def __init__(self, config):
        super(JointModel, self).__init__(config)

    def forward(self, batch, eval_visdial=False, eval_coref=False):
        # load data
        batch_size = len(batch['whole_dialog_index_flatten'])

        # get embedding and forward visdial
        tokens = batch['tokens']
        segments = batch['segments']
        sep_indices = batch['sep_indices'] 
        mask = batch['mask']
        hist_len = batch['hist_len']
        image_feat = batch['image_feat'] 
        image_loc = batch['image_loc'] 
        image_mask = batch['image_mask']
        next_sentence_labels = batch.get('next_sentence_labels', None)
        image_target = batch.get('image_target', None)
        image_label = batch.get('image_label', None)

        sequence_lengths = torch.gather(sep_indices, 1, hist_len.view(-1, 1)) + 1
        sequence_lengths = sequence_lengths.squeeze(1)
        attention_mask_lm_nsp = sequence_mask(sequence_lengths, max_len=tokens.shape[1])
        sep_len = hist_len + 1

        whole_dialog_index_flatten = batch['whole_dialog_index_flatten']
        losses = OrderedDict()
        if eval_visdial:
            # debug
            if self.config['debugging']:
                log.info('one batch feed eval visdial')

            num_lines = tokens.size(0)
            line_batch_size = self.config['eval_line_batch_size']
            num_line_batches = num_lines // line_batch_size
            if num_lines % line_batch_size > 0:
                num_line_batches += 1
            whole_dialog_index_flatten = batch['whole_dialog_index_flatten'].tolist()
            text_embedding = []
            nsp_scores = []
            for j in range(num_line_batches):
                # create chunks of the original batch
                chunk_range = range(j*line_batch_size, min((j+1)*line_batch_size, num_lines))
                tokens_chunk = tokens[chunk_range]
                segments_chunk = segments[chunk_range]
                sep_indices_chunk = sep_indices[chunk_range]
                mask_chunk = mask[chunk_range]
                sep_len_chunk = sep_len[chunk_range]
                attention_mask_lm_nsp_chunk = attention_mask_lm_nsp[chunk_range]
                image_feat_chunk = image_feat[chunk_range]
                image_loc_chunk = image_loc[chunk_range]
                image_mask_chunk = image_mask[chunk_range]

                text_embedding_chunk, _, _, _, nsp_scores_chunk = \
                    self.encoder(
                        tokens_chunk,
                        image_feat_chunk,
                        image_loc_chunk,
                        sep_indices=sep_indices_chunk,
                        sep_len=sep_len_chunk,
                        token_type_ids=segments_chunk,
                        masked_lm_labels=mask_chunk,
                        attention_mask=attention_mask_lm_nsp_chunk,
                        image_attention_mask=image_mask_chunk
                    )
                if len(whole_dialog_index_flatten) > 0 and whole_dialog_index_flatten[0] in chunk_range:
                    text_embedding.append(text_embedding_chunk[whole_dialog_index_flatten[0] - j*line_batch_size])
                    whole_dialog_index_flatten.pop(0)
                del text_embedding_chunk
                nsp_scores.append(nsp_scores_chunk)
            text_embedding = torch.stack(text_embedding, 0)
            whole_dialog_index_flatten = torch.arange(batch_size)
            nsp_scores = torch.cat(nsp_scores, 0)

            # debug
            if self.config['debugging']:
                log.info('one batch done eval visdial')

        elif eval_coref:
            # debug
            if self.config['debugging']:
                log.info('one batch feed eval coref')

            whole_dialog_index_flatten = batch['whole_dialog_index_flatten']
            tokens = tokens[whole_dialog_index_flatten]
            segments = segments[whole_dialog_index_flatten]
            sep_indices = sep_indices[whole_dialog_index_flatten]
            mask = mask[whole_dialog_index_flatten]
            sep_len = sep_len[whole_dialog_index_flatten]
            attention_mask_lm_nsp = attention_mask_lm_nsp[whole_dialog_index_flatten]
            image_feat = image_feat[whole_dialog_index_flatten]
            image_loc = image_loc[whole_dialog_index_flatten]
            image_mask = image_mask[whole_dialog_index_flatten]
            text_embedding, _, _, _, _ = \
                self.encoder(
                    tokens,
                    image_feat,
                    image_loc,
                    sep_indices=sep_indices,
                    sep_len=sep_len,
                    token_type_ids=segments,
                    masked_lm_labels=mask,
                    attention_mask=attention_mask_lm_nsp,
                    image_attention_mask=image_mask
                )
            whole_dialog_index_flatten = torch.arange(batch_size)
            nsp_scores = None

            # debug
            if self.config['debugging']:
                log.info('one batch done eval coref')


        else:
            # debug
            if self.config['debugging']:
                log.info('one batch forward')

            text_embedding, losses['lm_loss'], losses['img_loss'], losses['nsp_loss'], nsp_scores = \
                self.encoder(
                    tokens,
                    image_feat,
                    image_loc,
                    sep_indices=sep_indices,
                    sep_len=sep_len,
                    token_type_ids=segments,
                    masked_lm_labels=mask,
                    attention_mask=attention_mask_lm_nsp,
                    next_sentence_label=next_sentence_labels,
                    image_attention_mask=image_mask,
                    image_label=image_label,
                    image_target=image_target
                )

            text_embedding = text_embedding[whole_dialog_index_flatten]
            whole_dialog_index_flatten = torch.arange(batch_size)

            # debug
            if self.config['debugging']:
                log.info('one batch done')

        if self.config['train_on_dense']:
            if nsp_scores is not None:
                nsp_scores_output = nsp_scores.detach().clone()
                if not eval_visdial:
                    nsp_scores = nsp_scores.view(-1, self.config['num_options_dense'], 2)
                if 'next_sentence_labels' in batch and self.config['nsp_loss_coeff'] > 0:
                    losses['nsp_loss'] = F.cross_entropy(nsp_scores.view(-1,2), batch['next_sentence_labels'].view(-1)) 
                else:
                    losses['nsp_loss'] = None

                if not eval_visdial:
                    gt_relevance = batch['gt_relevance'].to(self.config['device'])
                    losses['ce_loss'] = self.dense_loss(F.log_softmax(nsp_scores[:, :, 0], dim=1), F.softmax(gt_relevance, dim=1))
                else:
                    losses['ce_loss'] = None
            else:
                nsp_scores_output = None
                losses['nsp_loss'] = None
                losses['ce_loss'] = None
        else:
            if nsp_scores is not None:
                nsp_scores_output = nsp_scores.detach()
            else:
                nsp_scores_output = None

        # train coref
        losses['coref_loss'] = 0
        if eval_visdial or eval_coref:
            coref_predictions = []
        else:
            coref_predictions = None
        for dialog_id in range(batch_size):
            whole_dialog_index = whole_dialog_index_flatten[dialog_id]
            text_embedding_cur = text_embedding[whole_dialog_index].unsqueeze(0) # [1, num_words, hidden]

            input_tensors = [
                batch['input_mask'][dialog_id],
                batch['speaker_ids'][dialog_id],
                batch['genre_id'][dialog_id],
                batch['candidate_starts'][dialog_id],
                batch['candidate_ends'][dialog_id],
                batch['cand_cluster_ids'][dialog_id],
            ]

            input_tensors = [t.to(self.config['device']) for t in input_tensors]
            input_tensors.insert(0, text_embedding_cur)

            (
                # [cand_num]
                cand_mention_scores,
                # [top_cand_num]
                top_start_idxes,
                # [top_cand_num]
                top_end_idxes,
                # [top_cand_num]
                top_span_cluster_ids,
                # [top_span_num, pruned_ant_num]
                top_ant_idxes_of_spans,
                # [top_cand_num, pruned_ant_num]
                top_ant_cluster_ids_of_spans,
                # # [top_cand_num, 1 + pruned_ant_num]
                top_ant_scores_of_spans,
                # [top_span_num, pruned_ant_num]
                top_ant_mask_of_spans,
                # [top_span_num, 1 + top_span_num], [top_span_num, top_span_num]
                full_fast_ant_scores_of_spans, full_ant_mask_of_spans
            ) = self.forward_coref(*input_tensors)

            losses['coref_loss'] += self.compute_coref_loss(
                # [cand_num]
                cand_mention_scores,
                # [top_cand_num]
                top_start_idxes,
                # [top_cand_num]
                top_end_idxes,
                # [top_cand_num]
                top_span_cluster_ids,
                # [top_span_num, pruned_ant_num]
                top_ant_idxes_of_spans,
                # [top_cand_num, pruned_ant_num]
                top_ant_cluster_ids_of_spans,
                # # [top_cand_num, 1 + pruned_ant_num]
                top_ant_scores_of_spans,
                # [top_span_num, pruned_ant_num]
                top_ant_mask_of_spans,
                # [top_span_num, 1 + top_span_num], [top_span_num, top_span_num]
                full_fast_ant_scores_of_spans, full_ant_mask_of_spans
            )

            if eval_visdial or eval_coref:
                predictions = self.predict_coref(
                    # [cand_num]0
                    cand_mention_scores,
                    # [top_cand_num]
                    top_start_idxes,
                    # [top_cand_num]
                    top_end_idxes,
                    # [top_cand_num]
                    top_span_cluster_ids,
                    # [top_span_num, pruned_ant_num]
                    top_ant_idxes_of_spans,
                    # [top_cand_num, pruned_ant_num]
                    top_ant_cluster_ids_of_spans,
                    # # [top_cand_num, 1 + pruned_ant_num]
                    top_ant_scores_of_spans,
                    # [top_span_num, pruned_ant_num]
                    top_ant_mask_of_spans
                )
                coref_predictions.append(predictions)
        losses['coref_loss'] /= batch_size

        output = {
            'losses': losses,
            'coref_predictions': coref_predictions,
            'nsp_scores': nsp_scores_output,
        }
        return output


class JointRunner(Runner):
    def __init__(self, config):
        super(JointRunner, self).__init__(config)
        self.model = JointModel(config)
        self.model.to(self.config['device'])

        if not self.config['validating'] or self.config['dp_type'] == 'apex':
            # self.optimizer = OptimizerBase.from_opt(self.model, self.config)
            self.optimizer, self.scheduler = init_optim(self.model, self.config)

    def forward(self, batch, eval_visdial=False, eval_coref=False):
        # load data
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.config['device'])

        output = self.model(batch, eval_visdial=eval_visdial, eval_coref=eval_coref)
        losses = output['losses']

        losses['visdial_loss'] = 0
        for key in ['lm_loss', 'img_loss', 'nsp_loss']:
            if key in losses and losses[key] is not None:
                losses[key] = losses[key].mean()
                losses['visdial_loss'] += self.config[f'{key}_coeff'] * losses[key]

        losses['tot_loss'] = 0
        for key in ['visdial_loss', 'coref_loss']:
            if losses[key] is not None:
                losses['tot_loss'] += self.config[f'{key}_coeff'] * losses[key]

        for key in losses:
            if (losses[key] is not None) and (key != 'tot_loss') and (isinstance(losses[key], torch.Tensor)):
                losses[key] = losses[key].detach()

        output['losses'] = losses

        return output        


class JointDenseRunner(Runner):
    def __init__(self, config):
        super(JointDenseRunner, self).__init__(config)
        self.model = JointModel(config)
        self.model.to(self.config['device'])

        self.dense_loss = nn.KLDivLoss(reduction='batchmean')

        if not self.config['validating'] or self.config['dp_type'] == 'apex':
            # self.optimizer = OptimizerBase.from_opt(self.model, self.config)
            self.optimizer, self.scheduler = init_optim(self.model, self.config)

    def forward(self, batch, eval_visdial=False, eval_coref=False):
        # load data
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.config['device'])
                
        output = self.model(batch, eval_visdial=eval_visdial, eval_coref=eval_coref)
        losses = output['losses']

        losses['visdial_loss'] = 0
        for key in ['nsp_loss', 'ce_loss']:
            if key in losses and losses[key] is not None:
                losses[key] = losses[key].mean()
                losses['visdial_loss'] += self.config[f'{key}_coeff'] * losses[key]

        losses['tot_loss'] = 0
        for key in ['visdial_loss', 'coref_loss']:
            if losses[key] is not None:
                losses['tot_loss'] += self.config[f'{key}_coeff'] * losses[key]

        for key in losses:
            if (losses[key] is not None) and (key != 'tot_loss') and (isinstance(losses[key], torch.Tensor)):
                losses[key] = losses[key].detach()

        output['losses'] = losses

        return output