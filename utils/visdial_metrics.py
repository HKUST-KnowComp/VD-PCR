"""
A Metric observes output of certain model, for example, in form of logits or
scores, and accumulates a particular metric with reference to some provided
targets. In context of VisDial, we use Recall (@ 1, 5, 10), Mean Rank, Mean
Reciprocal Rank (MRR) and Normalized Discounted Cumulative Gain (NDCG).

Each ``Metric`` must atleast implement three methods:
    - ``observe``, update accumulated metric with currently observed outputs
      and targets.
    - ``retrieve`` to return the accumulated metric., an optionally reset
      internally accumulated metric (this is commonly done between two epochs
      after validation).
    - ``reset`` to explicitly reset the internally accumulated metric.

Caveat, if you wish to implement your own class of Metric, make sure you call
``detach`` on output tensors (like logits), else it will cause memory leaks.
"""
import torch
import torch.distributed as dist
import numpy as np

def scores_to_ranks(scores: torch.Tensor):
    """Convert model output scores into ranks."""
    batch_size, num_rounds, num_options = scores.size()
    scores = scores.view(-1, num_options)

    # sort in descending order - largest score gets highest rank
    sorted_ranks, ranked_idx = scores.sort(1, descending=True)

    # i-th position in ranked_idx specifies which score shall take this
    # position but we want i-th position to have rank of score at that
    # position, do this conversion
    ranks = ranked_idx.clone().fill_(0)
    for i in range(ranked_idx.size(0)):
        for j in range(num_options):
            ranks[i][ranked_idx[i][j]] = j
    # convert from 0-99 ranks to 1-100 ranks
    ranks += 1
    ranks = ranks.view(batch_size, num_rounds, num_options)
    return ranks

class SparseGTMetrics(object):
    """
    A class to accumulate all metrics with sparse ground truth annotations.
    These include Recall (@ 1, 5, 10), Mean Rank and Mean Reciprocal Rank.
    """

    def __init__(self):
        self._rank_list = []
        self._rank_list_rnd = []
        self.num_rounds = None

    def observe(
        self, predicted_scores: torch.Tensor, target_ranks: torch.Tensor
    ):
        predicted_scores = predicted_scores.detach()

        # shape: (batch_size, num_rounds, num_options)
        predicted_ranks = scores_to_ranks(predicted_scores)
        batch_size, num_rounds, num_options = predicted_ranks.size()
        self.num_rounds = num_rounds
        # collapse batch dimension
        predicted_ranks = predicted_ranks.view(
            batch_size * num_rounds, num_options
        )

        # shape: (batch_size * num_rounds, )
        target_ranks = target_ranks.view(batch_size * num_rounds).long()

        # shape: (batch_size * num_rounds, )
        predicted_gt_ranks = predicted_ranks[
            torch.arange(batch_size * num_rounds), target_ranks
        ]
        self._rank_list.extend(list(predicted_gt_ranks.cpu().numpy()))
        
        predicted_gt_ranks_rnd = predicted_gt_ranks.view(batch_size, num_rounds)
        #  predicted gt ranks
        self._rank_list_rnd.append(predicted_gt_ranks_rnd.cpu().numpy())

    def retrieve(self, reset: bool = True):
        num_examples = len(self._rank_list)
        if num_examples > 0:
            # convert to numpy array for easy calculation.
            __rank_list = torch.tensor(self._rank_list).float()
            metrics = {
                "r@1": torch.mean((__rank_list <= 1).float()).item(),
                "r@5": torch.mean((__rank_list <= 5).float()).item(),
                "r@10": torch.mean((__rank_list <= 10).float()).item(),
                "mean": torch.mean(__rank_list).item(),
                "mrr": torch.mean(__rank_list.reciprocal()).item()
            }
            # add round metrics
            _rank_list_rnd = np.concatenate(self._rank_list_rnd)
            _rank_list_rnd = _rank_list_rnd.astype(float)
            r_1_rnd = np.mean(_rank_list_rnd <= 1, axis=0)
            r_5_rnd = np.mean(_rank_list_rnd <= 5, axis=0)
            r_10_rnd = np.mean(_rank_list_rnd <= 10, axis=0)
            mean_rnd = np.mean(_rank_list_rnd, axis=0)
            mrr_rnd = np.mean(np.reciprocal(_rank_list_rnd), axis=0)

            for rnd in range(1, self.num_rounds + 1):
                metrics["r_1" + "_round_" + str(rnd)] = r_1_rnd[rnd-1]
                metrics["r_5" + "_round_" + str(rnd)] = r_5_rnd[rnd-1]
                metrics["r_10" + "_round_" + str(rnd)] = r_10_rnd[rnd-1]
                metrics["mean" + "_round_" + str(rnd)] = mean_rnd[rnd-1]
                metrics["mrr" + "_round_" + str(rnd)] = mrr_rnd[rnd-1]
        else:
            metrics = {}

        if reset:
            self.reset()
        return metrics

    def reset(self):
        self._rank_list = []
        self._rank_list_rnd = []

class NDCG(object):
    def __init__(self):
        self._ndcg_numerator = 0.0
        self._ndcg_denominator = 0.0

    def observe(
        self, predicted_scores: torch.Tensor, target_relevance: torch.Tensor
    ):
        """
        Observe model output scores and target ground truth relevance and
        accumulate NDCG metric.

        Parameters
        ----------
        predicted_scores: torch.Tensor
            A tensor of shape (batch_size, num_options), because dense
            annotations are available for 1 randomly picked round out of 10.
        target_relevance: torch.Tensor
            A tensor of shape same as predicted scores, indicating ground truth
            relevance of each answer option for a particular round.
        """
        predicted_scores = predicted_scores.detach()

        # shape: (batch_size, 1, num_options)
        predicted_scores = predicted_scores.unsqueeze(1)
        predicted_ranks = scores_to_ranks(predicted_scores)

        # shape: (batch_size, num_options)
        predicted_ranks = predicted_ranks.squeeze(1)
        batch_size, num_options = predicted_ranks.size()
        
        k = torch.sum(target_relevance != 0, dim=-1)

        # shape: (batch_size, num_options)
        _, rankings = torch.sort(predicted_ranks, dim=-1)
        # Sort relevance in descending order so highest relevance gets top rnk.
        _, best_rankings = torch.sort(
            target_relevance, dim=-1, descending=True
        )

        # shape: (batch_size, )
        batch_ndcg = []
        for batch_index in range(batch_size):
            num_relevant = k[batch_index]
            dcg = self._dcg(
                rankings[batch_index][:num_relevant],
                target_relevance[batch_index],
            )
            best_dcg = self._dcg(
                best_rankings[batch_index][:num_relevant],
                target_relevance[batch_index],
            )
            batch_ndcg.append(dcg / best_dcg)

        self._ndcg_denominator += batch_size
        self._ndcg_numerator += sum(batch_ndcg)

    def _dcg(self, rankings: torch.Tensor, relevance: torch.Tensor):
        sorted_relevance = relevance[rankings].cpu().float()
        discounts = torch.log2(torch.arange(len(rankings)).float() + 2)
        return torch.sum(sorted_relevance / discounts, dim=-1)

    def retrieve(self, reset: bool = True):
        if self._ndcg_denominator > 0:
            metrics = {
                "ndcg": float(self._ndcg_numerator / self._ndcg_denominator)
            }
        else:
            metrics = {}

        if reset:
            self.reset()
        return metrics

    def reset(self):
        self._ndcg_numerator = 0.0
        self._ndcg_denominator = 0.0

class SparseGTMetricsParallel(object):
    """
    A class to accumulate all metrics with sparse ground truth annotations.
    These include Recall (@ 1, 5, 10), Mean Rank and Mean Reciprocal Rank.
    """

    def __init__(self, gpu_rank):
        self.rank_1 = 0
        self.rank_5 = 0
        self.rank_10 = 0
        self.ranks = 0
        self.reciprocal = 0
        self.count = 0
        self.gpu_rank = gpu_rank
        self.img_ids = []

    def observe(
        self, img_id: list, predicted_scores: torch.Tensor, target_ranks: torch.Tensor
    ):
        if img_id in self.img_ids:
            return
        else:
            self.img_ids.append(img_id)

        predicted_scores = predicted_scores.detach()

        # shape: (batch_size, num_rounds, num_options)
        predicted_ranks = scores_to_ranks(predicted_scores)
        batch_size, num_rounds, num_options = predicted_ranks.size()
        self.num_rounds = num_rounds
        # collapse batch dimension
        predicted_ranks = predicted_ranks.view(
            batch_size * num_rounds, num_options
        )

        # shape: (batch_size * num_rounds, )
        target_ranks = target_ranks.view(batch_size * num_rounds).long()

        # shape: (batch_size * num_rounds, )
        predicted_gt_ranks = predicted_ranks[
            torch.arange(batch_size * num_rounds), target_ranks
        ]

        self.rank_1 += (predicted_gt_ranks <= 1).sum().item()
        self.rank_5 += (predicted_gt_ranks <= 5).sum().item()
        self.rank_10 += (predicted_gt_ranks <= 10).sum().item()
        self.ranks += predicted_gt_ranks.sum().item()
        self.reciprocal += predicted_gt_ranks.float().reciprocal().sum().item()
        self.count += batch_size * num_rounds

    def retrieve(self):
        if self.count > 0:
            # retrieve data from all gpu
            # define tensor on GPU, count and total is the result at each GPU
            t = torch.tensor([self.rank_1, self.rank_5, self.rank_10, self.ranks, self.reciprocal, self.count], dtype=torch.float32, device=f'cuda:{self.gpu_rank}')
            dist.barrier()  # synchronizes all processes
            dist.all_reduce(t, op=torch.distributed.ReduceOp.SUM,)  # Reduces the tensor data across all machines in such a way that all get the final result.
            t = t.tolist()
            self.rank_1, self.rank_5, self.rank_10, self.ranks, self.reciprocal, self.count = t

            # convert to numpy array for easy calculation.
            metrics = {
                "r@1": self.rank_1 / self.count,
                "r@5": self.rank_5 / self.count,
                "r@10": self.rank_10 / self.count,
                "mean": self.ranks / self.count,
                "mrr": self.reciprocal / self.count,
                "tot_rnds": self.count,
            }

        else:
            metrics = {}

        return metrics

    def get_count(self):
        return int(self.count)

class NDCGParallel(NDCG):
    def __init__(self, gpu_rank):
        super(NDCGParallel, self).__init__()
        self.gpu_rank = gpu_rank
        self.img_ids = []
        self.count = 0

    def observe(
        self, img_id: int, predicted_scores: torch.Tensor, target_relevance: torch.Tensor
    ):
        """
        Observe model output scores and target ground truth relevance and
        accumulate NDCG metric.

        Parameters
        ----------
        predicted_scores: torch.Tensor
            A tensor of shape (batch_size, num_options), because dense
            annotations are available for 1 randomly picked round out of 10.
        target_relevance: torch.Tensor
            A tensor of shape same as predicted scores, indicating ground truth
            relevance of each answer option for a particular round.
        """
        if img_id in self.img_ids:
            return
        else:
            self.img_ids.append(img_id)
            self.count += 1

        super(NDCGParallel, self).observe(predicted_scores, target_relevance)


    def retrieve(self):
        if self._ndcg_denominator > 0:
            # define tensor on GPU, count and total is the result at each GPU
            t = torch.tensor([self._ndcg_numerator, self._ndcg_denominator, self.count], dtype=torch.float32, device=f'cuda:{self.gpu_rank}')
            dist.barrier()  # synchronizes all processes
            dist.all_reduce(t, op=torch.distributed.ReduceOp.SUM,)  # Reduces the tensor data across all machines in such a way that all get the final result.
            t = t.tolist()
            self._ndcg_numerator, self._ndcg_denominator, self.count = t
            metrics = {
                "ndcg": float(self._ndcg_numerator / self._ndcg_denominator)
            }
        else:
            metrics = {}
        return metrics

    def get_count(self):
        return int(self.count)
