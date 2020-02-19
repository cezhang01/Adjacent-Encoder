import numpy as np


class Data():

    def __init__(self, args):

        self.parse_args(args)
        self.load_data()

    def parse_args(self, args):

        self.dataset_name = args.dataset_name
        self.x = args.x
        self.trans_induc = args.trans_induc
        self.minibatch_size = args.minibatch_size
        if self.trans_induc == 'transductive':
            self.training_ratio = 1
        elif self.trans_induc == 'inductive':
            self.training_ratio = args.training_ratio - args.training_ratio * 0.1
            self.validation_ratio = args.training_ratio * 0.1
            self.test_ratio = 1 - args.training_ratio

    def load_data(self):

        self.doc = self.doc_preprocessing(np.loadtxt('./cora/' + self.dataset_name + '/content.txt'))
        self.num_doc = len(self.doc)
        self.label = np.loadtxt('./cora/' + self.dataset_name + '/label.txt')
        self.adjacency_matrix = self.generate_symmetric_adjacency_matrix(np.loadtxt('./cora/' + self.dataset_name + '/adjacency_matrix.txt'))
        self.links = self.generate_links(self.adjacency_matrix)
        np.random.shuffle(self.links)
        self.voc = np.genfromtxt('./cora/' + self.dataset_name + '/voc.txt', dtype=str)
        self.num_tokens = len(self.voc)

        if self.trans_induc == 'transductive':
            self.doc_training, self.doc_test = self.doc, self.doc
            self.label_training, self.label_test = self.label, self.label
            self.adjacency_matrix_training, self.adjacency_matrix_test = self.adjacency_matrix, self.adjacency_matrix
            self.links_training = self.links
        elif self.trans_induc == 'inductive':
            self.doc_training, self.doc_test = self.doc[:int(self.num_doc * self.training_ratio)], self.doc[int(self.num_doc * (self.training_ratio + self.validation_ratio)):]
            self.label_training, self.label_test = self.label[:int(self.num_doc * self.training_ratio)], self.label[int(self.num_doc * (self.training_ratio + self.validation_ratio)):]
            self.adjacency_matrix_training, self.adjacency_matrix_test = \
                self.adjacency_matrix[:int(self.num_doc * self.training_ratio)][:int(self.num_doc * self.training_ratio)], \
                self.adjacency_matrix[int(self.num_doc * (self.training_ratio + self.validation_ratio)):][:int(self.num_doc * self.training_ratio)]
            self.links_training, self.links_test = self.split_links(self.links)

        self.num_minibatch = int(np.ceil(len(self.links_training) / self.minibatch_size))
        self.minibatch_data = {}
        for minibatch_index in range(1, self.num_minibatch + 1):
            self.prepare_minibatch(num_minibatch=self.num_minibatch, minibatch_index=minibatch_index)
            self.minibatch_data[minibatch_index] = {}
            self.minibatch_data[minibatch_index]['sampling_links'] = self.sampling_links
            self.minibatch_data[minibatch_index]['neighbor_ids'] = self.neighbor_ids
            self.minibatch_data[minibatch_index]['segment_ids'] = self.segment_ids

        if self.x == 0:
            self.input_training = self.doc_training
            self.input_test = self.doc_test
        elif self.x == 1:
            self.input_training = np.concatenate([self.doc_training, self.adjacency_matrix_training], axis=1)
            self.input_test = np.concatenate([self.doc_test, self.adjacency_matrix_test], axis=1)

    def doc_preprocessing(self, doc):

        doc_preprocessed = []
        for row in doc:
            max_row = np.log(1 + np.max(row))
            doc_preprocessed.append(np.log(1 + row) / max_row)

        return np.asarray(doc_preprocessed)

    def generate_symmetric_adjacency_matrix(self, adjacency_matrix):

        adjacency_matrix_symm = np.zeros((len(adjacency_matrix), len(adjacency_matrix)))
        for row_idx in range(len(adjacency_matrix)):
            for col_idx in range(len(adjacency_matrix)):
                if adjacency_matrix[row_idx, col_idx] == 1:
                    adjacency_matrix_symm[row_idx, col_idx] = 1
                    adjacency_matrix_symm[col_idx, row_idx] = 1
                if row_idx == col_idx:
                    adjacency_matrix_symm[row_idx, col_idx] = 1

        return adjacency_matrix_symm

    def generate_links(self, adjacency_matrix):

        links = []
        for row_idx in range(len(adjacency_matrix)):
            for col_idx in range(len(adjacency_matrix)):
                if adjacency_matrix[row_idx, col_idx] == 1:
                    links.append([row_idx, col_idx])

        return np.asarray(links)

    def split_links(self, links):

        links_training, links_test = [], []
        for link in links:
            if link[0] < int(self.num_doc * self.training_ratio) and link[1] < int(self.num_doc * self.training_ratio):
                links_training.append([link[0], link[1]])
            elif link[0] >= int(self.num_doc * (self.training_ratio + self.validation_ratio)) and link[1] >= int(self.num_doc * (self.training_ratio + self.validation_ratio)):
                links_test.append([link[0], link[1]])

        return np.asarray(links_training), np.asarray(links_test)

    def prepare_minibatch(self, num_minibatch, minibatch_index):

        self.sampling_links = self.sample_minibatch_links(num_minibatch, minibatch_index)
        self.neighbor_ids, self.segment_ids = self.prepare_neighbors()

    def sample_minibatch_links(self, num_minibatch, minibatch_index):

        if minibatch_index == num_minibatch:
            sampling_links = self.links_training[self.minibatch_size * (minibatch_index - 1):]
            if self.minibatch_size - len(sampling_links) != 0:
                indices = np.random.choice(len(self.links_training), self.minibatch_size - len(sampling_links), replace=False)
                sampling_links = np.concatenate((sampling_links, self.links_training[indices]), axis=0)
        else:
            sampling_links = self.links_training[self.minibatch_size * (minibatch_index - 1):self.minibatch_size * minibatch_index]

        return sampling_links

    def prepare_neighbors(self):

        neighbor_ids, segment_ids = [], []
        for idx, link in enumerate(self.sampling_links):
            neighbor_ids_tmp = self.links_training[self.links_training[:, 0] == link[0]][:, 1]
            neighbor_ids.extend(neighbor_ids_tmp)
            segment_ids += len(neighbor_ids_tmp) * [idx]

        return np.asarray(neighbor_ids), np.asarray(segment_ids)