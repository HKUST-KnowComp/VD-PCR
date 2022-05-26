## Hyperparameters in VD-PCR ##

The hyperparameters of Phase 0, Phase 1, and Phase 2 are saved in `conly.conf` (only pronoun coreference training), `joint.conf` (joint training), `vonly.conf` (only visual dialog training), respectively.

### basic
| name | description | default value |
| ---- | ---- | ---- |
| use_coref | to use pronoun coreference in training and evaluation | `True` for `conly` and `joint`, `False` for `vonly` |
| train_on_dense | to train on VisDial dense annotations | `True` for Phase 2, `False` otherwise |
| metrics_to_maximize | the metric to select the best model | `prp_f1` for Phase 0, `mrr` for Phase 1, `ndcg` for Phase 2 |

### data
| name | description | default value |
| ---- | ---- | ---- |
| visdial_image_feats | directory of the extracted features of ViLBERT | `data/visdial/visdial_img_feat.lmdb` |
| visdial_[SPLIT] | VisDial data| | 
| visdial_[SPLIT]_dense | VisDial data with dense annotations | | 
| visdial_[SPLIT]\_dense_annotations | VisDial dense annotations | |
| rlv_hst_[SPLIT] | relevant history for history pruning | history of `crf_cap` rules |
| start_path | parameters to load if `loads_start_path=True` | |
| model_config | bert model config | |
| dataloader_text_only | only load text in dataloader for debugging | `False` |
| rlv_hst_only | to include relevant history only | `True` for Phase 2, `False` otherwise |
| rlv_hst_dense_round | if rlv_hst files only contain relevant history on the rounds with dense annotations | `False` |

### visdial training
| name | description | default value |
| ---- | ---- | ---- |
| visdial_tot_rounds | #rounds in a VisDial dialogue | 11 |
| num_negative_samples | #negative samples for each positive sample in Phase 1 training | 1 |
| sequences_per_image | #samples for each VisDial image and dialogue | 2 |
| batch_size | batch size for training | 4 |
| [TASK]\_loss_coeff | loss coefficients for tasks | |
| batch_multiply | to conduct gradient backward propagation for every #batches | 10 for Phase 2, 1 otherwise |
| use_trainval | to train on VisDial train and val set | `False` |
| dense_loss | loss with dense_annotations in Phase 2 training | options: `ce` (cross entropy), `listmle`, `listnet`, or `approxndcg`; default: `ce` |
| coref only | only pronoun coreference training | `True` for Phase 0, `False` otherwise |

### visdial model
| name | description | default value |
| ---- | ---- | ---- |
| mask_prob | probability to mask text tokens for LM task | 0.1 |
| image_mask_prob | probability to image text tokens for LM task | 0.1 |
| max_seq_len | maximum length of input sequence | 256 |
| num_options | #answer options to use in VisDial | 100 | 
| num_options_dense | #answer options to use in Phase 2 | 100 | 
| use_embedding | embedding for model | `vilbert` for `conly` and `vonly`, `joint` or `joint_head` for `joint` |

### visdial evaluation
| name | description | default value |
| ---- | ---- | ---- |
| eval_coref_on_test | to evaluate coreference on test set in joint training | `False` |
| eval_visdial_on_test | to evaluate VisDial on test set in joint training | `False` |
| eval_batch_size | batch size for evaluation | 1 |
| eval_line_batch_size | #lines to feed in each forward propagation | 40 |
| skip_mrr_eval | to skip evaluation of retrieval metrics | `True` for Phase 2, `False` otherwise |
| skip_eval | to skip evaluation during training | `True` for Phase 2 trainval, `False` otherwise |
| continue_evaluation | to continue the aborted evaluation | `False` |
| eval_at_start | to conduct evaluation at the start of training | `False` |

## restore ckpt
| name | description | default value |
| ---- | ---- | ---- |
| loads_best_ckpt | to load the best checkpoint in the model dir | `False` |
| loads_ckpt | to load the last checkpoint in the model dir | `False` |
| restarts | to restart training disregarding the training so far | `False` |
| resets_max_metric | to reset the maximum metric disregarding the metrics so far | `False` |
| uses_new_optimizer | to use a new optimizer disregarding the saved optimizer | `False` |
| sets_new_lr | to set a new learning rate disregarding the saved learning rate | `False` |
| loads_start_path | to load `start_path` for model initialization | `False` |

### training
| name | description | default value |
| ---- | ---- | ---- |
| random_seed | random seed | 2020 |
| next_logging_pct | to log the process for every percentage of data | 1.0 |
| next_evaluating_pct | to evaluate the model for every percentage of data | 50.0 |
| max_ckpt_to_keep | #checkpoints to keep in the model dir; past checkpoints would to deleted automatically while the best checkpoint is always kept | 1 |
| num_epochs | #epochs for training | 10 |
| early_stop_epoch | to early stop after `metrics_to_maximize` does not improve for #epochs | 3 |
| skip_saving_ckpt | to skip checkpoint saving for quick debugging | `False` |
| dp_type | data parallel type | `ddp` or `apex` (ddp with apex) for Phase 1, `dp` for Phase 2 |
| stop_epochs | to stop after training for #epochs | 3 for Phase 2, -1 otherwise |

### predicting
| name | description | default value |
| ---- | ---- | ---- |
| predicting_split | split to predict on | |
| predict_shards_num | to divide the data into #shards for parallel prediction | |
| predict_shard | the shard index to predict on for this GPU during parallel prediction | |
| predict_each_round | to predict pronoun coreference separately for each dialogue round | `False` |
| predict_dense_round | to predict pronoun coreference only for dialogue rounds with dense annotations (because prediction on all data takes too long) | `False` |
| num_test_dialogs | #dialogues in test data | 8000 | 
| num_val_dialogs | #dialogues in val data | 2064 |
| save_score | to save scores for ensembling | `True` for ensembling, `False` otherwise |

### optimizer
| name | description | default value |
| ---- | ---- | ---- |
| reset_optim | to reset part of the optimizer disregarding the saved one | options: `none` (load everything from the checkpoint), `all`, `states`, `keep_states`; default: `none` |
| learning_rate_bert | learning rate for ViLBERT base model | |
| learning_rate_task | learning rate for task layers | |
| min_lr | minimum learning rate in learning rate schedule | |
| decay_method_* | learning rate decay method | `linear` |
| decay_exp | the decay exponent for `exp` decay method | 2 |
| max_grad_norm | to clip gradient to this value if it > 0 | 0 |
| task_optimizer | optimizer for task layers | options: `adam` or `adamw`; default: `adam` |
| warmup_ratio | to warm up learning rate for #ratio of the whole training steps | 0.1 |

### directory
| name | description | default value |
| ---- | ---- | ---- |
| log_dir | log directory to save the model directory | |
| data_dir | data directory | |
| visdial_output_dir | directory of visdial prediction in the model directory | |
| coref_output_dir | directory of coreference prediction in the model directory | |
| bert_cache_dir | directory to save `transformers` model cache | |

### coref data
| name | description | default value |
| ---- | ---- | ---- |
| id_to_genre | genre id in coreferencd model | |