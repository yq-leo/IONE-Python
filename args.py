from argparse import ArgumentParser


def make_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='phone-email',
                        choices=['phone-email', 'ACM-DBLP', 'foursquare-twitter'],
                        help='available datasets: phone-email, ACM-DBLP, foursquare-twitter')
    parser.add_argument('--total_iter', dest='total_iter', type=int, default=10000000, help='total iteration')
    parser.add_argument('--out_dim', dest='out_dim', type=int, default=100, help='dimension of output embeddings')
    parser.add_argument('--gpu', dest='device', action='store_const', const='cuda', default='cpu', help='use GPU')

    return parser.parse_args()
