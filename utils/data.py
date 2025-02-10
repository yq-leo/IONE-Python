import numpy as np


def load_dataset(dataset, p=0.2):
    """
    Load dataset.
    :param dataset: dataset name
    :param p: training ratio
    :return:
        edge_index1, edge_index2: edge list of graph G1, G2
        anchor_links: training node alignments, i.e., anchor links
        test_pairs: test node alignments
    """

    data = np.load(f'{dataset}_{p:.1f}.npz')
    edge_index1, edge_index2 = data['edge_index1'].T.astype(np.int64), data['edge_index2'].T.astype(np.int64)
    anchor_links, test_pairs = data['pos_pairs'].astype(np.int64), data['test_pairs'].astype(np.int64)

    return edge_index1, edge_index2, anchor_links, test_pairs
