import numpy as np
from collections import Counter
from scipy.optimize import linear_sum_assignment as linear_assignment
import torch
import torch.distributed as dist

def f1(p_num, p_den, r_num, r_den, beta=1):
    p = 0 if p_den == 0 else p_num / float(p_den)
    r = 0 if r_den == 0 else r_num / float(r_den)
    return 0 if p + r == 0 else (1 + beta * beta) * p * r / (beta * beta * p + r)

class CorefEvaluator(object):
    def __init__(self):
        self.evaluators = [Evaluator(m) for m in (muc, b_cubed, ceafe)]

    def update(self, predicted, gold, mention_to_predicted, mention_to_gold):
        for e in self.evaluators:
            e.update(predicted, gold, mention_to_predicted, mention_to_gold)

    def get_f1(self):
        return sum(e.get_f1() for e in self.evaluators) / len(self.evaluators)

    def get_recall(self):
        return sum(e.get_recall() for e in self.evaluators) / len(self.evaluators)

    def get_precision(self):
        return sum(e.get_precision() for e in self.evaluators) / len(self.evaluators)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

class Evaluator(object):
    def __init__(self, metric, beta=1):
        self.p_num = 0
        self.p_den = 0
        self.r_num = 0
        self.r_den = 0
        self.metric = metric
        self.beta = beta

    def update(self, predicted, gold, mention_to_predicted, mention_to_gold):
        if self.metric == ceafe:
            pn, pd, rn, rd = self.metric(predicted, gold)
        else:
            pn, pd = self.metric(predicted, mention_to_gold)
            rn, rd = self.metric(gold, mention_to_predicted)
        self.p_num += pn
        self.p_den += pd
        self.r_num += rn
        self.r_den += rd

    def get_f1(self):
        return f1(self.p_num, self.p_den, self.r_num, self.r_den, beta=self.beta)

    def get_recall(self):
        return 0 if self.r_num == 0 else self.r_num / float(self.r_den)

    def get_precision(self):
        return 0 if self.p_num == 0 else self.p_num / float(self.p_den)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

    def get_counts(self):
        return self.p_num, self.p_den, self.r_num, self.r_den


def evaluate_documents(documents, metric, beta=1):
    evaluator = Evaluator(metric, beta=beta)
    for document in documents:
        evaluator.update(document)
    return evaluator.get_precision(), evaluator.get_recall(), evaluator.get_f1()


def b_cubed(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:
        if len(c) == 1:
            continue

        gold_counts = Counter()
        correct = 0
        for m in c:
            if m in mention_to_gold:
                gold_counts[tuple(mention_to_gold[m])] += 1
        for c2, count in gold_counts.items():
            if len(c2) != 1:
                correct += count * count

        num += correct / float(len(c))
        dem += len(c)

    return num, dem


def muc(clusters, mention_to_gold):
    tp, p = 0, 0
    for c in clusters:
        p += len(c) - 1
        tp += len(c)
        linked = set()
        for m in c:
            if m in mention_to_gold:
                linked.add(mention_to_gold[m])
            else:
                tp -= 1
        tp -= len(linked)
    return tp, p


def phi4(c1, c2):
    return 2 * len([m for m in c1 if m in c2]) / float(len(c1) + len(c2))


def ceafe(clusters, gold_clusters):
    clusters = [c for c in clusters if len(c) != 1]
    scores = np.zeros((len(gold_clusters), len(clusters)))
    for i in range(len(gold_clusters)):
        for j in range(len(clusters)):
            scores[i, j] = phi4(gold_clusters[i], clusters[j])
    matching = linear_assignment(-scores)
    similarity = scores[matching[0], matching[1]].sum()
    return similarity, len(clusters), similarity, len(gold_clusters)


def lea(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:
        if len(c) == 1:
            continue

        common_links = 0
        all_links = len(c) * (len(c) - 1) / 2.0
        for i, m in enumerate(c):
            if m in mention_to_gold:
                for m2 in c[i + 1:]:
                    if m2 in mention_to_gold and mention_to_gold[m] == mention_to_gold[m2]:
                        common_links += 1

        num += len(c) * common_links / float(all_links)
        dem += len(c)

    return num, dem


def verify_correct_NP_match(predicted_NP, gold_NPs, model, matched_gold_ids):
    if model == 'exact':
        for gold_id, tmp_gold_NP in enumerate(gold_NPs):
            if gold_id in matched_gold_ids:
                continue
            if tmp_gold_NP[0] == predicted_NP[0] and tmp_gold_NP[1] == predicted_NP[1]:
                return gold_id
    elif model == 'cover':
        for gold_id, tmp_gold_NP in enumerate(gold_NPs):
            if gold_id in matched_gold_ids:
                continue
            if tmp_gold_NP[0] <= predicted_NP[0] and tmp_gold_NP[1] >= predicted_NP[1]:
                return gold_id
            if tmp_gold_NP[0] >= predicted_NP[0] and tmp_gold_NP[1] <= predicted_NP[1]:
                return gold_id
    return None
    

class PrCorefEvaluator(object):
    def __init__(self):
        self.all_coreference = 0
        self.predict_coreference = 0
        self.correct_predict_coreference = 0
        self.pronoun_list = ['she', 'her', 'he', 'him', 'them', 'they', 'She', 'Her', 'He', 'Him', 'Them', 'They', 'it', 'It', 'his', 'hers', 'its', 'their', 'theirs', 'His', 'Hers', 'Its', 'Their', 'Theirs']

    def get_prf(self):
        p = 0 if self.predict_coreference == 0 else self.correct_predict_coreference / self.predict_coreference
        r = 0 if self.all_coreference == 0 else self.correct_predict_coreference / self.all_coreference
        f1 = 0 if p + r == 0 else 2 * p * r / (p + r)

        return p, r, f1


    def update(self, predicted_clusters, pronoun_info, sentences):
        predicted_clusters = [tuple(pc) for pc in predicted_clusters]

        for pronoun_example in pronoun_info:
            tmp_pronoun_index = pronoun_example['current_pronoun'][0]

            tmp_candidate_NPs = pronoun_example['candidate_NPs']
            tmp_correct_candidate_NPs = pronoun_example['correct_NPs']

            find_pronoun = False
            for coref_cluster in predicted_clusters:
                for mention in coref_cluster:
                    mention_start_index = mention[0]
                    if mention_start_index == tmp_pronoun_index:
                        find_pronoun = True
                if find_pronoun and pronoun_example['reference_type'] == 0:
                    matched_cdd_np_ids = []
                    matched_crr_np_ids = []
                    for mention in coref_cluster:
                        mention_start_index = mention[0]
                        tmp_mention_span = (
                            mention_start_index,
                            mention[1])
                        matched_np_id = verify_correct_NP_match(tmp_mention_span, tmp_candidate_NPs, 'cover', matched_cdd_np_ids)
                        if matched_np_id is not None:
                            # exclude such scenario: predict 'its' and overlap with candidate 'its eyes'
                            # predict +1 but correct +0
                            if tmp_mention_span[0] < len(sentences) and\
                                tmp_mention_span[0] == tmp_mention_span[1] and\
                                sentences[tmp_mention_span[0]] in self.pronoun_list and\
                                len(tmp_candidate_NPs[matched_np_id]) > 1:
                                continue
                            matched_cdd_np_ids.append(matched_np_id)
                            self.predict_coreference += 1
                            matched_np_id = verify_correct_NP_match(tmp_mention_span, tmp_correct_candidate_NPs, 'cover', matched_crr_np_ids)
                            if matched_np_id is not None:
                                matched_crr_np_ids.append(matched_np_id)
                                self.correct_predict_coreference += 1
                    break

            self.all_coreference += len(tmp_correct_candidate_NPs)


def gather_round_metrics(coref_evaluator):
    coref_precision, coref_recall, coref_f1 = list(zip(*[c.get_prf()
                                                        for c in coref_evaluator]))
    coref_precision_rnds = np.mean(coref_precision)
    coref_recall_rnds = np.mean(coref_recall)
    coref_f1_rnds = np.mean(coref_f1)
    coref_precision, coref_recall, coref_f1 = coref_precision[-1], coref_recall[-1], coref_f1[-1]
    return (coref_precision_rnds, coref_recall_rnds, coref_f1_rnds), \
        (coref_precision, coref_recall, coref_f1)


def prf(p_num, p_den, r_num, r_den, beta=1):
    p = 0 if p_den == 0 else p_num / float(p_den)
    r = 0 if r_den == 0 else r_num / float(r_den)
    f = 0 if p + r == 0 else (1 + beta * beta) * p * r / (beta * beta * p + r)
    return p, r, f
