import random
import torch.nn.functional as F

from utils.data import load_dataset
from utils.metrics import *
from args import make_args


class IONE:
    def __init__(self, data):
        self.data = data

    def get_network_anchors(self):
        answer_map_x, answer_map_y = dict(), dict()
        g1, g2 = self.data['g1'], self.data['g2']
        for anchor_link in self.data['anchor_links']:
            answer_map_x[f'{anchor_link[0]}_{g1}'] = f'{anchor_link[1]}_{g2}'
            answer_map_y[f'{anchor_link[1]}_{g2}'] = f'{anchor_link[0]}_{g1}'
        return answer_map_x, answer_map_y

    def train(self, total_iter, dim):
        two_order_x = IONEUpdate(self.data['g1'], self.data['edges1'], dim)
        two_order_x.init()

        two_order_y = IONEUpdate(self.data['g2'], self.data['edges2'], dim)
        two_order_y.init()

        anchor_x, anchor_y = self.get_network_anchors()

        print(f'number of anchors: {len(anchor_x)}')
        print(f'number of anchors: {len(anchor_y)}')

        for i in range(total_iter):
            two_order_x.train(i=i,
                              iter_count=total_iter,
                              two_order_answer=two_order_x.answer,
                              two_order_answer_context_input=two_order_x.answer_context_input,
                              two_order_answer_context_output=two_order_x.answer_context_output,
                              anchors=anchor_x)
            two_order_y.train(i=i,
                              iter_count=total_iter,
                              two_order_answer=two_order_x.answer,
                              two_order_answer_context_input=two_order_x.answer_context_input,
                              two_order_answer_context_output=two_order_x.answer_context_output,
                              anchors=anchor_y)

        n1 = len(two_order_x.answer)
        n2 = len(two_order_y.answer)
        embs_x = torch.vstack([two_order_x.answer[f"{uid}_{self.data['g1']}"] for uid in range(n1)])
        embs_y = torch.vstack([two_order_y.answer[f"{uid}_{self.data['g2']}"] for uid in range(n2)])
        return embs_x, embs_y


class IONEUpdate:
    def __init__(self, g, edges, out_dim):
        self.g = g
        self.edges = edges
        self.dimension = out_dim

        self.vertex = {}

        self.answer = {}
        self.answer_context_input = {}
        self.answer_context_output = {}

        self.source_id = []
        self.target_id = []
        self.edge_weight = []
        self.alias = []
        self.prob = []
        self.neg_table = []

        self.init_rho = 0.025
        self.rho = 0
        self.num_negative = 5
        self.neg_table_size = 10000000

    def read_data(self):
        for edge in self.edges:
            source = f'{edge[0]}_{self.g}'
            target = f'{edge[1]}_{self.g}'
            self.source_id.append(source)
            self.target_id.append(target)

            weight = float(1)
            self.edge_weight.append(weight)

            if source in self.vertex:
                self.vertex[source] += weight
            else:
                self.vertex[source] = weight
                self.answer[source] = torch.rand(self.dimension, dtype=torch.float32) * (1.0 / self.dimension) - (0.5 / self.dimension)
                self.answer_context_input[source] = torch.zeros(self.dimension, dtype=torch.float32)
                self.answer_context_output[source] = torch.zeros(self.dimension, dtype=torch.float32)

            if target in self.vertex:
                self.vertex[target] += weight
            else:
                self.vertex[target] = weight
                self.answer[target] = torch.rand(self.dimension, dtype=torch.float32) * (1.0 / self.dimension) - (0.5 / self.dimension)
                self.answer_context_input[target] = torch.zeros(self.dimension, dtype=torch.float32)
                self.answer_context_output[target] = torch.zeros(self.dimension, dtype=torch.float32)

    def init_alias_table(self):
        large_block = []
        small_block = []

        sum_weights = sum(self.edge_weight)
        norm_prob = [w * len(self.edge_weight) / sum_weights for w in self.edge_weight]

        for i in range(len(self.edge_weight) - 1, -1, -1):
            if norm_prob[i] < 1:
                small_block.append(i)
            else:
                large_block.append(i)
        num_small_block = len(small_block)
        num_large_block = len(large_block)

        self.prob = [0] * len(self.edge_weight)
        self.alias = [0] * len(self.edge_weight)

        while num_small_block > 0 and num_large_block > 0:
            num_small_block = num_small_block - 1
            cur_small_block = small_block[num_small_block]
            num_large_block = num_large_block - 1
            cur_large_block = large_block[num_large_block]

            self.prob[cur_small_block] = norm_prob[cur_small_block]
            self.alias[cur_small_block] = cur_large_block

            norm_prob[cur_large_block] = norm_prob[cur_large_block] + norm_prob[cur_small_block] - 1
            if norm_prob[cur_large_block] < 1:
                small_block[num_small_block] = cur_large_block
                num_small_block = num_small_block + 1
            else:
                large_block[num_large_block] = cur_large_block
                num_large_block = num_large_block + 1

        while num_large_block > 0:
            num_large_block = num_large_block - 1
            self.prob[large_block[num_large_block]] = 1

        while num_small_block > 0:
            num_small_block = num_small_block - 1
            self.prob[small_block[num_small_block]] = 1

    def sample_edge(self, rand1: float, rand2: float) -> int:
        k = int(len(self.edge_weight) * rand1)
        return k if rand2 < self.prob[k] else self.alias[k]

    def init_neg_table(self):
        self.neg_table = []
        total_sum = sum([v ** 0.75 for v in self.vertex.values()])

        cumulative_sum = 0
        por = 0
        vertex_keys = list(self.vertex.keys())
        iter_keys = iter(vertex_keys)
        current_key = next(iter_keys)

        for i in range(self.neg_table_size):
            if (i + 1) / self.neg_table_size > por:
                cumulative_sum += self.vertex[current_key] ** 0.75
                por = cumulative_sum / total_sum
                if por >= 1:
                    self.neg_table.append(current_key)
                    continue
                if i != 0:
                    current_key = next(iter_keys)
            self.neg_table.append(current_key)

    def update(self, vec_u, vec_v, vec_error, label, source, target, two_order_answer, two_order_answer_context,
               anchors):
        if source in anchors:
            vec_u = two_order_answer[anchors[source]] if anchors[source] in two_order_answer else two_order_answer[source]
        if target in anchors:
            vec_v = two_order_answer_context[anchors[target]] if anchors[target] in two_order_answer_context else two_order_answer_context[target]

        x = vec_u @ vec_v
        g = (label - torch.sigmoid(x)) * self.rho

        vec_error += g * vec_v
        if target in anchors:
            if anchors[target] not in two_order_answer_context:
                vec_v += g * vec_u
            else:
                two_order_answer_context[anchors[target]] += g * vec_u
        else:
            vec_v += g * vec_u

    def update_reverse(self, vec_u, vec_v, vec_error, label, source, target, two_order_answer, two_order_answer_context,
                       anchors):
        vec_error = torch.zeros(self.dimension, dtype=torch.float32)
        if source in anchors:
            vec_u = two_order_answer[anchors[source]] if anchors[source] in two_order_answer else two_order_answer[source]
        if target in anchors:
            vec_v = two_order_answer_context[anchors[target]] if anchors[target] in two_order_answer_context else two_order_answer_context[target]

        x = vec_u @ vec_v
        g = (label - torch.sigmoid(x)) * self.rho

        vec_error += g * vec_v
        if target in anchors:
            if anchors[target] not in two_order_answer_context:
                vec_v += g * vec_u
            else:
                two_order_answer_context[anchors[target]] += g * vec_u
        else:
            vec_v += g * vec_u

        uid_1 = source
        if uid_1 in anchors:
            if anchors[uid_1] not in two_order_answer:
                vec_u = two_order_answer[uid_1]
                self.answer[uid_1] += vec_error
            else:
                two_order_answer[anchors[uid_1]] += vec_error
        else:
            self.answer[uid_1] += vec_error

    def train(self, i, iter_count, two_order_answer, two_order_answer_context_input, two_order_answer_context_output,
              anchors):
        vec_error = torch.zeros(self.dimension, dtype=torch.float32)
        vec_error_reverse = torch.zeros(self.dimension, dtype=torch.float32)
        if i % int(iter_count/10) == 0:
            self.rho = self.init_rho * (1.0 - i / iter_count)
            if self.rho < self.init_rho * 0.0001:
                self.rho = self.init_rho * 0.0001
            print(f"{i} {self.rho}")

        edge_id = self.sample_edge(random.random(), random.random())
        uid_1 = self.source_id[edge_id]
        uid_2 = self.target_id[edge_id]

        d = 0
        while d < self.num_negative + 1:
            if d == 0:
                label = 1
                target = uid_2
            else:
                neg_index = random.randint(0, self.neg_table_size - 1)
                target = self.neg_table[neg_index]
                if target == uid_1 or target == uid_2:
                    continue
                label = 0

            self.update(vec_u=self.answer[uid_1],
                        vec_v=self.answer_context_input[target],
                        vec_error=vec_error,
                        label=label,
                        source=uid_1,
                        target=target,
                        two_order_answer=two_order_answer,
                        two_order_answer_context=two_order_answer_context_input,
                        anchors=anchors)
            self.update_reverse(vec_u=self.answer[target],
                                vec_v=self.answer_context_output[uid_1],
                                vec_error=vec_error_reverse,
                                label=label,
                                source=target,
                                target=uid_1,
                                two_order_answer=two_order_answer,
                                two_order_answer_context=two_order_answer_context_output,
                                anchors=anchors)
            d = d + 1

        if uid_1 in anchors:
            vec_u = two_order_answer[anchors[uid_1]] if anchors[uid_1] in two_order_answer else None
            if vec_u is None:
                vec_u = two_order_answer[uid_1]
                self.answer[uid_1] += vec_error
            else:
                two_order_answer[anchors[uid_1]] += vec_error

        else:
            self.answer[uid_1] += vec_error

    def output(self, output_filename: str):
        with open(output_filename, 'w') as file:
            for uid, vector in self.answer.items():
                file.write(f"{uid} {'|'.join(map(str, vector))}\n")

    def init(self):
        self.read_data()
        self.init_alias_table()
        self.init_neg_table()


if __name__ == "__main__":
    args = make_args()

    g1, g2 = args.dataset.split("-")
    edge_index1, edge_index2, anchor_links, test_pairs = load_dataset(f'datasets/{args.dataset}', p=0.2)
    data = {
        'g1': g1,
        'g2': g2,
        'edges1': edge_index1,
        'edges2': edge_index2,
        'anchor_links': anchor_links,
        'test_pairs': test_pairs
    }

    test = IONE(data)
    embs_x, embs_y = test.train(args.total_iter, args.out_dim)

    emb_x = F.normalize(embs_x, p=2, dim=1)
    emb_y = F.normalize(embs_y, p=2, dim=1)

    similarity = emb_x @ emb_y.T

    hits_ks_ltr = hits_ks_ltr_scores(similarity, torch.from_numpy(test_pairs), ks=[1, 5, 10, 30])
    mrr_ltr = mrr_ltr_score(similarity, torch.from_numpy(test_pairs))
    print("-------LTR-------")
    for k, hits_k in hits_ks_ltr.items():
        print(f"Hits@{k}: {hits_k:.4f}")
    print(f"MRR: {mrr_ltr:.4f}")
    print('-----------------')

    hits_ks_rtl = hits_ks_rtl_scores(similarity, torch.from_numpy(test_pairs), ks=[1, 5, 10, 30])
    mrr_rtl = mrr_rtl_score(similarity, torch.from_numpy(test_pairs))
    print("-------RTL-------")
    for k, hits_k in hits_ks_rtl.items():
        print(f"Hits@{k}: {hits_k:.4f}")
    print(f"MRR: {mrr_rtl:.4f}")
    print('-----------------')

    print("-------MAX-------")
    hits_ks = hits_ks_max_scores(similarity, torch.from_numpy(test_pairs), ks=[1, 5, 10, 30])
    mrr = mrr_max_score(similarity, torch.from_numpy(test_pairs))
    for k, hits_k in hits_ks.items():
        print(f"Hits@{k}: {hits_k:.4f}")
    print(f"MRR: {mrr:.4f}")
    print('-----------------')

    print("-------MEAN-------")
    hits_ks = hits_ks_mean_scores(similarity, torch.from_numpy(test_pairs), ks=[1, 5, 10, 30])
    mrr = mrr_mean_score(similarity, torch.from_numpy(test_pairs))
    for k, hits_k in hits_ks.items():
        print(f"Hits@{k}: {hits_k:.4f}")
    print(f"MRR: {mrr:.4f}")
    print('-----------------')


