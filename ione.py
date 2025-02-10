import numpy as np
import random

import torch

from utils.metrics import hits_ks_scores, mrr_score


class IONE:
    def __init__(self, foldtrain):
        self.foldtrain = foldtrain

    def get_network_anchors(self, postfix_1: str, postfix_2: str):
        answer_map = dict()
        file_name = f'AcrossNetworkEmbeddingData/twitter_foursquare_groundtruth/groundtruth.{self.foldtrain}.foldtrain.train.number'
        with open(file_name, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                answer_map[line + postfix_1] = line + postfix_2
        return answer_map

    def train(self, total_iter, file_interop, dim):
        networkx_file = f"AcrossNetworkEmbeddingData/foursquare/following{file_interop}"
        networky_file = f"AcrossNetworkEmbeddingData/twitter/following{file_interop}"
        output_filename_networkx = f"AcrossNetworkEmbeddingData/foursquare/embeddings/foursquare.embedding.update.2SameAnchor.{self.foldtrain}.foldtrain.twodirectionContext{file_interop}"
        output_filename_networky = f"AcrossNetworkEmbeddingData/twitter/embeddings/twitter.embedding.update.2SameAnchor.{self.foldtrain}.foldtrain.twodirectionContext{file_interop}"

        two_order_x = IONEUpdate(dim, networkx_file, "foursquare")
        two_order_x.init()

        two_order_y = IONEUpdate(dim, networky_file, "twitter")
        two_order_y.init()

        anchor_x = self.get_network_anchors("_foursquare", "_twitter")
        anchor_y = self.get_network_anchors("_twitter", "_foursquare")

        print(f'number of anchors: {len(anchor_x)}')
        print(f'number of anchors: {len(anchor_y)}')

        for i in range(total_iter):
            # if i % 10000 == 9999:
            #     print(f"Iteration {i + 1}")
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

        two_order_x.output(f"{output_filename_networkx}.{dim}_dim.{total_iter}")
        two_order_y.output(f"{output_filename_networky}.{dim}_dim.{total_iter}")
        n1 = len(two_order_x.answer)
        n2 = len(two_order_y.answer)
        embs_x = np.vstack([two_order_x.answer[f"{uid}_foursquare"] for uid in range(n1)])
        embs_y = np.vstack([two_order_y.answer[f"{uid}_twitter"] for uid in range(n2)])
        return embs_x, embs_y


class IONEUpdate:
    def __init__(self, dimension: int, filename: str, postfix: str):
        self.vertex = {}

        self.answer = {}
        self.answer_context_input = {}
        self.answer_context_output = {}

        self.source_id = []
        self.target_id = []
        self.edge_weight = []
        self.alias = []
        self.prob = []

        self.dimension = dimension

        self.neg_table = []
        self.sigmoid_table_size = 1000
        self.sigmoid_table = np.zeros(self.sigmoid_table_size).astype(np.float32)
        self.SIGMOID_BOUND = 6

        self.init_rho = 0.025
        self.rho = 0
        self.num_negative = 5
        self.neg_table_size = 10000000

        self.input_file = filename
        self.postfix = postfix
        self.rnd = random.Random(123)

    def read_data(self):
        with open(self.input_file, 'r') as file:
            count = 0
            for line in file:
                array = line.strip().split()
                source = array[0] + "_" + self.postfix
                target = array[1] + "_" + self.postfix
                self.source_id.append(source)
                self.target_id.append(target)

                weight = float(1)
                self.edge_weight.append(weight)

                if source in self.vertex:
                    self.vertex[source] += weight
                else:
                    self.vertex[source] = weight
                    self.answer[source] = np.random.uniform(-0.5 / self.dimension, 0.5 / self.dimension, self.dimension).astype(np.float32)
                    self.answer_context_input[source] = np.zeros(self.dimension).astype(np.float32)
                    self.answer_context_output[source] = np.zeros(self.dimension).astype(np.float32)

                if target in self.vertex:
                    self.vertex[target] += weight
                else:
                    self.vertex[target] = weight
                    self.answer[target] = np.random.uniform(-0.5 / self.dimension, 0.5 / self.dimension, self.dimension).astype(np.float32)
                    self.answer_context_input[target] = np.zeros(self.dimension).astype(np.float32)
                    self.answer_context_output[target] = np.zeros(self.dimension).astype(np.float32)

                if count % 10000 == 0:
                    print(f"Reading Edges {count}")
                count += 1
            print(f"Number of vertices: {len(self.vertex)}, Number of Edges: {count}")

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

    def init_sigmoid_table(self):
        for i in range(self.sigmoid_table_size):
            x = 2 * self.SIGMOID_BOUND * i / self.sigmoid_table_size - self.SIGMOID_BOUND
            self.sigmoid_table[i] = 1 / (1 + np.exp(-x))

    def fast_sigmoid(self, x: float):
        if x > self.SIGMOID_BOUND:
            return 1
        elif x < -self.SIGMOID_BOUND:
            return 0
        k = int((x + self.SIGMOID_BOUND) * self.sigmoid_table_size / self.SIGMOID_BOUND / 2)
        return self.sigmoid_table[k]

    def update(self, vec_u, vec_v, vec_error, label, source, target, two_order_answer, two_order_answer_context,
               anchors):
        if source in anchors:
            vec_u = two_order_answer[anchors[source]] if anchors[source] in two_order_answer else two_order_answer[source]
        if target in anchors:
            vec_v = two_order_answer_context[anchors[target]] if anchors[target] in two_order_answer_context else two_order_answer_context[target]

        x = vec_u @ vec_v
        g = (label - self.fast_sigmoid(x)) * self.rho

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
        vec_error = np.zeros(self.dimension).astype(np.float32)
        if source in anchors:
            vec_u = two_order_answer[anchors[source]] if anchors[source] in two_order_answer else two_order_answer[source]
        if target in anchors:
            vec_v = two_order_answer_context[anchors[target]] if anchors[target] in two_order_answer_context else two_order_answer_context[target]

        x = vec_u @ vec_v
        g = (label - self.fast_sigmoid(x)) * self.rho

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
        vec_error = np.zeros(self.dimension).astype(np.float32)
        vec_error_reverse = np.zeros(self.dimension).astype(np.float32)
        if i % 1000000 == 0:
            self.rho = self.init_rho * (1.0 - i / iter_count)
            if self.rho < self.init_rho * 0.0001:
                self.rho = self.init_rho * 0.0001
            print(f"{i} {self.rho}")

        edge_id = self.sample_edge(random.random(), random.random())
        uid_1 = self.source_id[edge_id]
        uid_2 = self.target_id[edge_id]

        label = 0
        d = 0
        while d < self.num_negative + 1:
            if d == 0:
                label = 1
                target = uid_2
            else:
                neg_index = random.randint(0, self.neg_table_size - 1)
                target = self.neg_table[neg_index]
                # if uid_1 is None or uid_2 is None or target is None:
                #     print(uid_1)
                #     print(uid_2)
                #     print(neg_index)
                #     print(target)
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
        self.init_sigmoid_table()


def load_test_data(foldtrain=9):
    test_file = f"AcrossNetworkEmbeddingData/twitter_foursquare_groundtruth/groundtruth.{foldtrain}.foldtrain.test.number"
    test_pairs = []
    with open(test_file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            test_pairs.append([int(line), int(line)])
    return np.array(test_pairs)


if __name__ == "__main__":
    trainfold = 2

    test_pairs = load_test_data(trainfold)
    test = IONE(trainfold)
    total_iter = int(1e7)
    embs_x, embs_y = test.train(total_iter, ".number", 100)

    emb_x = embs_x / (np.linalg.norm(embs_x, axis=1, keepdims=True) + 1e-10)
    emb_y = embs_y / (np.linalg.norm(embs_y, axis=1, keepdims=True) + 1e-10)

    similarity = np.dot(emb_x, emb_y.T)
    hits_ks = hits_ks_scores(torch.from_numpy(similarity), torch.from_numpy(test_pairs), ks=[1, 5, 10, 30])
    mrr = mrr_score(torch.from_numpy(similarity), torch.from_numpy(test_pairs))
    for k, hits_k in hits_ks.items():
        print(f"Hits@{k}: {hits_k:.4f}")
    print(f"MRR: {mrr:.4f}")
