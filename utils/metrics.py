import torch


def hits_ks_ltr_scores(similarity, test_pairs, ks=None):
    """
    Compute hits@k scores for a given list of k.
    :param similarity: similarity matrix (n1 x n2)
    :param test_pairs: test pairs
    :param ks: list of k
    :return:
        hits_ks: a dictionary of hits@k scores (k: hits@k)
    """

    hits_ks = {}
    ranks1 = torch.argsort(-similarity[test_pairs[:, 0]], dim=1)
    signal1_hit = ranks1 == test_pairs[:, 1].view(-1, 1)
    for k in ks:
        hits_ks[k] = torch.sum(signal1_hit[:, :k]) / test_pairs.shape[0]

    return hits_ks


def mrr_ltr_score(similarity, test_pairs):
    """
    Compute MRR scores.
    :param similarity: similarity matrix (n1 x n2)
    :param test_pairs: test pairs
    :return:
        mrr: MRR score
    """

    ranks1 = torch.argsort(-similarity[test_pairs[:, 0]], dim=1)
    signal1_hit = ranks1 == test_pairs[:, 1].view(-1, 1)
    mrr = torch.mean(1 / (torch.where(signal1_hit)[1].float() + 1))

    return mrr


def hits_ks_scores(similarity, test_pairs, ks=None):
    """
    Compute hits@k scores for a given list of k.
    :param similarity: similarity matrix (n1 x n2)
    :param test_pairs: test pairs
    :param ks: list of k
    :return:
        hits_ks: a dictionary of hits@k scores (k: hits@k)
    """

    hits_ks = {}

    ranks1 = torch.argsort(-similarity[test_pairs[:, 0]], dim=1)
    ranks2 = torch.argsort(-similarity.T[test_pairs[:, 1]], dim=1)

    signal1_hit = ranks1 == test_pairs[:, 1].view(-1, 1)
    signal2_hit = ranks2 == test_pairs[:, 0].view(-1, 1)
    for k in ks:
        hits_ltr = torch.sum(signal1_hit[:, :k]) / test_pairs.shape[0]
        hits_rtl = torch.sum(signal2_hit[:, :k]) / test_pairs.shape[0]
        hits_ks[k] = torch.max(hits_ltr, hits_rtl)

    return hits_ks


def mrr_score(similarity, test_pairs):
    """
    Compute MRR scores.
    :param similarity: similarity matrix (n1 x n2)
    :param test_pairs: test pairs
    :return:
        mrr: MRR score
    """

    ranks1 = torch.argsort(-similarity[test_pairs[:, 0]], dim=1)
    ranks2 = torch.argsort(-similarity.T[test_pairs[:, 1]], dim=1)

    signal1_hit = ranks1 == test_pairs[:, 1].view(-1, 1)
    signal2_hit = ranks2 == test_pairs[:, 0].view(-1, 1)

    mrr_ltr = torch.mean(1 / (torch.where(signal1_hit)[1].float() + 1))
    mrr_rtl = torch.mean(1 / (torch.where(signal2_hit)[1].float() + 1))
    mrr = torch.max(mrr_ltr, mrr_rtl)

    return mrr
