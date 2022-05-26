import copy
import json
from collections import OrderedDict
import torch
import torch.nn.functional as F
from torch import nn
import glog as log

import sys
sys.path.append('../')

from utils.data_utils import sequence_mask
from utils.optim_utils import OptimizerBase, init_optim
from utils.model_utils import listMLE, approxNDCGLoss, listNet
from models.vilbert_dialog import BertForMultiModalPreTraining, BertConfig
from models.vilbert_dialog_head import BertForMultiModalPreTrainingHead, BertConfigHead
from models.runner import Runner

class VisualDialogEncoder(nn.Module):

    def __init__(self, config_path, device, use_apex=False, cache_dir=None):
        super(VisualDialogEncoder, self).__init__()
        config = BertConfig.from_json_file(config_path)

        self.bert_pretrained = BertForMultiModalPreTraining.from_pretrained('bert-base-uncased', config, device, use_apex=use_apex, cache_dir=cache_dir)
        self.bert_pretrained.train()

    def forward(self, input_ids, image_feat, image_loc, sep_indices=None, sep_len=None, token_type_ids=None,
         attention_mask=None, masked_lm_labels=None, next_sentence_label=None,
         image_attention_mask=None, image_label=None, image_target=None):                      
        masked_lm_loss = None
        masked_img_loss = None
        nsp_loss = None
        seq_relationship_score = None

        if next_sentence_label is not None and masked_lm_labels \
            is not None and image_target is not None:
            # train mode, output losses
            masked_lm_loss, masked_img_loss, nsp_loss, _, _, seq_relationship_score  = \
                self.bert_pretrained(input_ids, image_feat, image_loc, sep_indices=sep_indices, sep_len=sep_len, \
                 token_type_ids=token_type_ids, attention_mask=attention_mask, masked_lm_labels=masked_lm_labels, \
                            next_sentence_label=next_sentence_label, image_attention_mask=image_attention_mask,\
                                image_label=image_label, image_target=image_target)
        else:
            #inference, output scores
            _, _, seq_relationship_score, _, _ = \
                self.bert_pretrained(input_ids, image_feat, image_loc, sep_indices=sep_indices, sep_len=sep_len, \
                    token_type_ids=token_type_ids, attention_mask=attention_mask, masked_lm_labels=masked_lm_labels, \
                    next_sentence_label=next_sentence_label, image_attention_mask=image_attention_mask,\
                    image_label=image_label, image_target=image_target)

        out = (masked_lm_loss, masked_img_loss, nsp_loss, seq_relationship_score)

        return out


class VisualDialogEmbedding(nn.Module):

    def __init__(self, config_path, device, use_apex=False, cache_dir=None):
        super(VisualDialogEmbedding, self).__init__()
        config = BertConfig.from_json_file(config_path)

        self.bert_pretrained = BertForMultiModalPreTraining.from_pretrained('bert-base-uncased', config, device, use_apex=use_apex, cache_dir=cache_dir)
        self.bert_pretrained.train()

    def forward(self, input_ids, image_feat, image_loc, sep_indices=None, sep_len=None, token_type_ids=None, attention_mask=None, image_attention_mask=None):

        text_embedding = self.bert_pretrained.get_text_embedding(input_ids, image_feat, image_loc, sep_indices=sep_indices, sep_len=sep_len, token_type_ids=token_type_ids, attention_mask=attention_mask, image_attention_mask=image_attention_mask, output_all_attention_masks=False)
        return (text_embedding, )


class VisualDialogHeadEmbedding(nn.Module):

    def __init__(self, config_path, device, use_apex=False, cache_dir=None):
        super(VisualDialogHeadEmbedding, self).__init__()
        config = BertConfigHead.from_json_file(config_path)

        self.bert_pretrained = BertForMultiModalPreTrainingHead.from_pretrained('bert-base-uncased', config, device, use_apex=use_apex, cache_dir=cache_dir)
        self.bert_pretrained.train()

    def forward(self, input_ids, image_feat, image_loc, sep_indices=None, sep_len=None, token_type_ids=None, attention_mask=None, image_attention_mask=None):

        text_embedding = self.bert_pretrained.get_text_embedding(input_ids, image_feat, image_loc, sep_indices=sep_indices, sep_len=sep_len, token_type_ids=token_type_ids, attention_mask=attention_mask, image_attention_mask=image_attention_mask, output_all_attention_masks=False)
        return (text_embedding, )


class VisualDialogEncoderWithEmbedding(nn.Module):

    def __init__(self, config_path, device, use_apex=False, cache_dir=None):
        super(VisualDialogEncoderWithEmbedding, self).__init__()
        config = BertConfig.from_json_file(config_path)

        self.bert_pretrained = BertForMultiModalPreTraining.from_pretrained('bert-base-uncased', config, device, use_apex=use_apex, cache_dir=cache_dir)
        self.bert_pretrained.train()

    def forward(self, input_ids, image_feat, image_loc, sep_indices=None, sep_len=None, token_type_ids=None,
         attention_mask=None, masked_lm_labels=None, next_sentence_label=None,
         image_attention_mask=None,image_label=None, image_target=None):                      

        masked_lm_loss = None
        masked_img_loss = None
        nsp_loss = None
        seq_relationship_score = None

        if next_sentence_label is not None and masked_lm_labels \
            is not None and image_target is not None:
            # train mode, output losses
            masked_lm_loss, masked_img_loss, nsp_loss, sequence_output_t, _, seq_relationship_score  = \
                self.bert_pretrained(input_ids, image_feat, image_loc, sep_indices=sep_indices, sep_len=sep_len, \
                 token_type_ids=token_type_ids, attention_mask=attention_mask, masked_lm_labels=masked_lm_labels, \
                            next_sentence_label=next_sentence_label, image_attention_mask=image_attention_mask,\
                                image_label=image_label, image_target=image_target)
        else:
            #inference, output scores
            _, _, seq_relationship_score, sequence_output_t, _ = \
                self.bert_pretrained(input_ids, image_feat, image_loc, sep_indices=sep_indices, sep_len=sep_len, \
                    token_type_ids=token_type_ids, attention_mask=attention_mask, masked_lm_labels=masked_lm_labels, \
                    next_sentence_label=next_sentence_label, image_attention_mask=image_attention_mask,\
                    image_label=image_label, image_target=image_target)

        out = (sequence_output_t, masked_lm_loss, masked_img_loss, nsp_loss, seq_relationship_score)

        return out


class VisualDialogEncoderWithHeadEmbedding(nn.Module):

    def __init__(self, config_path, device, use_apex=False, cache_dir=None):
        super(VisualDialogEncoderWithHeadEmbedding, self).__init__()
        config = BertConfigHead.from_json_file(config_path)

        self.bert_pretrained = BertForMultiModalPreTrainingHead.from_pretrained('bert-base-uncased', config, device, use_apex=use_apex, cache_dir=cache_dir)
        self.bert_pretrained.train()

    def forward(self, input_ids, image_feat, image_loc, sep_indices=None, sep_len=None, token_type_ids=None,
         attention_mask=None, masked_lm_labels=None, next_sentence_label=None,
         image_attention_mask=None,image_label=None, image_target=None):                      

        masked_lm_loss = None
        masked_img_loss = None
        nsp_loss = None
        seq_relationship_score = None

        if next_sentence_label is not None and masked_lm_labels \
            is not None and image_target is not None:
            # train mode, output losses
            masked_lm_loss, masked_img_loss, nsp_loss, sequence_output_t, _, seq_relationship_score  = \
                self.bert_pretrained(input_ids, image_feat, image_loc, sep_indices=sep_indices, sep_len=sep_len, \
                 token_type_ids=token_type_ids, attention_mask=attention_mask, masked_lm_labels=masked_lm_labels, \
                            next_sentence_label=next_sentence_label, image_attention_mask=image_attention_mask,\
                                image_label=image_label, image_target=image_target)
        else:
            #inference, output scores
            _, _, seq_relationship_score, sequence_output_t, _ = \
                self.bert_pretrained(input_ids, image_feat, image_loc, sep_indices=sep_indices, sep_len=sep_len, \
                    token_type_ids=token_type_ids, attention_mask=attention_mask, masked_lm_labels=masked_lm_labels, \
                    next_sentence_label=next_sentence_label, image_attention_mask=image_attention_mask,\
                    image_label=image_label, image_target=image_target)

        out = (sequence_output_t, masked_lm_loss, masked_img_loss, nsp_loss, seq_relationship_score)

        return out


class VisdialRunner(Runner):
    def __init__(self, config):
        super(VisdialRunner, self).__init__(config)
        self.model = VisualDialogEncoder(self.config['model_config'], self.config['device'], 
                                         use_apex=self.config['dp_type'] == 'apex', 
                                         cache_dir=self.config['bert_cache_dir'])
        self.model.to(self.config['device'])

        if not self.config['validating'] or self.config['dp_type'] == 'apex':
            # self.optimizer = OptimizerBase.from_opt(self.model, self.config)
            self.optimizer, self.scheduler = init_optim(self.model, self.config)

    def forward(self, batch, eval_visdial=False, eval_coref=False):
        # load data
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.config['device'])

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

        losses = OrderedDict()

        if eval_visdial:
            # debug
            if self.config['debugging']:
                log.info('feed one batch eval')

            num_lines = tokens.size(0)
            line_batch_size = self.config['eval_line_batch_size']
            num_line_batches = num_lines // line_batch_size
            if num_lines % line_batch_size > 0:
                num_line_batches += 1
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

                _, _, _, nsp_scores_chunk = \
                    self.model(
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
                nsp_scores.append(nsp_scores_chunk)
            nsp_scores = torch.cat(nsp_scores, 0).detach()

            # debug
            if self.config['debugging']:
                log.info('one batch done eval')

        else:
            # debug
            if self.config['debugging']:
                log.info('feed one batch')

            losses['lm_loss'], losses['img_loss'], losses['nsp_loss'], _ = \
                self.model(
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
            # debug
            if self.config['debugging']:
                log.info('one batch done')

            nsp_scores = None

        losses['tot_loss'] = 0
        for key in ['lm_loss', 'img_loss', 'nsp_loss']:
            if key in losses and losses[key] is not None:
                losses[key] = losses[key].mean()
                losses['tot_loss'] += self.config[f'{key}_coeff'] * losses[key]

        output = {
            'losses': losses,
            'nsp_scores': nsp_scores
            }
        return output

class VisdialDenseRunner(Runner):
    def __init__(self, config):
        super(VisdialDenseRunner, self).__init__(config)
        self.model = VisualDialogEncoder(self.config['model_config'], self.config['device'], 
                                         use_apex=self.config['dp_type'] == 'apex', 
                                         cache_dir=self.config['bert_cache_dir'])
        if not(self.config['parallel'] and self.config['dp_type'] == 'dp'):
            self.model.to(self.config['device'])

        if self.config['dense_loss'] == 'ce':
            self.dense_loss = nn.KLDivLoss(reduction='batchmean')
        elif self.config['dense_loss'] == 'listmle':
            self.dense_loss = listMLE
        elif self.config['dense_loss'] == 'listnet':
            self.dense_loss = listNet
        elif self.config['dense_loss'] == 'approxndcg':
            self.dense_loss = approxNDCGLoss
        else:
            raise ValueError('dense_loss must be one of ce, listmle, listnet, approxndcg')

        if not self.config['validating'] or self.config['dp_type'] == 'apex':
            # self.optimizer = OptimizerBase.from_opt(self.model, self.config)
            self.optimizer, self.scheduler = init_optim(self.model, self.config)

    def forward(self, batch, eval_visdial=False, eval_coref=False):
        # load data
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.config['device'])

        # get embedding and forward visdial
        tokens = batch['tokens']
        segments = batch['segments']
        sep_indices = batch['sep_indices'] 
        mask = batch['mask']
        hist_len = batch['hist_len']
        image_feat = batch['image_feat'] 
        image_loc = batch['image_loc'] 
        image_mask = batch['image_mask']

        sequence_lengths = torch.gather(sep_indices, 1, hist_len.view(-1, 1)) + 1
        sequence_lengths = sequence_lengths.squeeze(1)
        attention_mask_lm_nsp = sequence_mask(sequence_lengths, max_len=tokens.shape[1])
        sep_len = hist_len + 1

        losses = OrderedDict()

        if eval_visdial:
            num_lines = tokens.size(0)
            line_batch_size = self.config['eval_line_batch_size']
            num_line_batches = num_lines // line_batch_size
            if num_lines % line_batch_size > 0:
                num_line_batches += 1
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

                _, _, _, nsp_scores_chunk = \
                    self.model(
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
                nsp_scores.append(nsp_scores_chunk)
            nsp_scores = torch.cat(nsp_scores, 0)

        else:
            _, _, _, nsp_scores = \
                self.model(
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

        if nsp_scores is not None:
            nsp_scores_output = nsp_scores.detach().clone()
            if not eval_visdial:
                nsp_scores = nsp_scores.view(-1, self.config['num_options_dense'], 2)
            if 'next_sentence_labels' in batch and self.config['nsp_loss_coeff'] > 0:
                next_sentence_labels = batch['next_sentence_labels'].to(self.config['device'])
                losses['nsp_loss'] = F.cross_entropy(nsp_scores.view(-1,2), next_sentence_labels.view(-1)) 
            else:
                losses['nsp_loss'] = None

            if not eval_visdial:
                gt_relevance = batch['gt_relevance'].to(self.config['device'])
                nsp_scores = nsp_scores[:, :, 0]
                if self.config['dense_loss'] == 'ce':
                    losses['dense_loss'] = self.dense_loss(F.log_softmax(nsp_scores, dim=1), F.softmax(gt_relevance, dim=1))
                else:
                    losses['dense_loss'] = self.dense_loss(nsp_scores, gt_relevance)
            else:
                losses['dense_loss'] = None
        else:
            nsp_scores_output = None
            losses['nsp_loss'] = None
            losses['dense_loss'] = None

        losses['tot_loss'] = 0
        for key in ['nsp_loss', 'dense_loss']:
            if key in losses and losses[key] is not None:
                losses[key] = losses[key].mean()
                losses['tot_loss'] += self.config[f'{key}_coeff'] * losses[key]

        output = {
            'losses': losses,
            'nsp_scores': nsp_scores_output
            }

        return output        