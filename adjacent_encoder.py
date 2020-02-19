import tensorflow as tf
import numpy as np
import time
from classification import classification_knn
from clustering import clustering_kmeans
from parametric_tsne import ParametricTSNE


class adjacent_encoder():

    def __init__(self, args, data):

        self.parse_args(args, data)
        self.show_config()
        self.generate_placeholders()
        self.generate_variables()

    def parse_args(self, args, data):

        self.data = data
        self.dataset_name = args.dataset_name
        self.num_doc = self.data.num_doc
        self.num_links = len(self.data.links)
        self.num_training_links = len(self.data.links_training)
        self.num_labels = int(max(self.data.label))
        self.tokens = self.data.num_tokens

        self.learning_rate = args.learning_rate
        self.num_epoch = args.num_epoch
        self.trans_induc = args.trans_induc
        self.x = args.x
        self.training_ratio = args.training_ratio
        self.minibatch_size = args.minibatch_size
        self.num_topics = args.num_topics

        self.sigma = args.sigma
        self.contractive = args.contractive
        self.sparsity = args.sparsity
        if self.sparsity == 0:
            self.topk = self.num_topics
            self.alpha = 0
        else:
            self.topk = args.topk
            self.alpha = 0#args.alpha

    def show_config(self):

        print('******************************************************')
        print('dataset name:', self.dataset_name)
        print('#documents:', self.num_doc)
        print('#links:', self.num_links)
        print('#tokens:', self.data.num_tokens)
        print('#labels:', self.num_labels)

        print('learning rate:', self.learning_rate)
        print('#epoch:', self.num_epoch)
        print('trans_induc:', self.trans_induc)
        print('X:', self.x)
        print('training ratio:', self.training_ratio)
        print('minibatch size:', self.minibatch_size)
        print('#topics:', self.num_topics)

        print('sigma:', self.sigma)
        print('contractive:', self.contractive)
        print('topk:', self.topk)
        print('******************************************************')

    def generate_placeholders(self):

        self.sampling_links = tf.placeholder('int32', [self.minibatch_size, 2])
        self.doc = tf.placeholder('float64', [None, len(self.data.input_training[0])])
        self.neighbor_ids = tf.placeholder('int32', [None])
        self.segment_ids = tf.placeholder('int32', [None])
        self.sm = tf.placeholder('float64', [])

    def generate_variables(self):

        self.weights = {
            'encoder_w': tf.Variable(tf.random_normal([len(self.data.input_training[0]), self.num_topics], dtype='float64'), dtype='float64'),
        }
        self.biases = {
            'encoder_b': tf.Variable(tf.random_normal([self.num_topics], dtype='float64'), dtype='float64'),
            'decoder_b': tf.Variable(tf.random_normal([len(self.data.input_training[0])], dtype='float64'), dtype='float64'),
        }
        self.att_w = tf.Variable(tf.random_normal([self.num_topics, self.num_topics], dtype='float64'), dtype='float64')
        self.att_a = tf.Variable(tf.random_normal([2 * self.num_topics], dtype='float64'), dtype='float64')

    def add_noise(self):

        if self.sigma != 0:
            self.doc_noisy = self.doc + tf.random_normal([tf.shape(self.doc)[0], tf.shape(self.doc)[1]], stddev=self.sm, dtype='float64')
        else:
            self.doc_noisy = self.doc

        return self.doc_noisy

    def encoder(self):

        self.doc_embed = tf.nn.tanh(tf.add(tf.matmul(self.doc_noisy, self.weights['encoder_w']), self.biases['encoder_b']))

        return self.doc_embed

    def neighbor_competition(self):

        sampling_doc_embed = tf.gather(self.doc_embed, self.sampling_links[:, 0])
        sampling_doc_embed_repeat = tf.gather(sampling_doc_embed, self.segment_ids)
        neighbor_doc_embed = tf.gather(self.doc_embed, self.neighbor_ids)
        attention = tf.reduce_sum(tf.multiply(sampling_doc_embed_repeat, neighbor_doc_embed), axis=1)  # inner-product attention
        # attention = tf.squeeze(tf.nn.relu(tf.matmul(tf.concat([tf.matmul(sampling_doc_embed_repeat, self.att_w), tf.matmul(neighbor_doc_embed, self.att_w)], axis=1), tf.expand_dims(self.att_a, 1))))  # self attention

        self.attention_norm = []
        for idx in range(self.minibatch_size):
            if idx == 0:
                self.attention_norm = tf.nn.softmax(tf.boolean_mask(attention, tf.equal(self.segment_ids, idx)))
            else:
                self.attention_norm = tf.concat([self.attention_norm, tf.nn.softmax(tf.boolean_mask(attention, tf.equal(self.segment_ids, idx)))], axis=0)
        embed_tmp = tf.multiply(neighbor_doc_embed, tf.tile(tf.expand_dims(self.attention_norm, 1), [1, self.num_topics]))
        self.sampling_doc_embed_aggregate = tf.segment_sum(embed_tmp, self.segment_ids)

        return self.sampling_doc_embed_aggregate

    def sparse_encoding(self):

        # the code of this function, sparse_encoding, is borrowed from https://github.com/hugochan/KATE
        # reference: Chen, Y., and Zaki, M. J. 2017. Kate: K-competitive autoencoder for text. In Proceedings of the ACM SIGKDD International Conference on Data Mining and Knowledge Discovery.

        P = (self.sampling_doc_embed_aggregate + tf.abs(self.sampling_doc_embed_aggregate)) / 2
        N = (self.sampling_doc_embed_aggregate - tf.abs(self.sampling_doc_embed_aggregate)) / 2

        values, indices = tf.nn.top_k(P, int(self.topk / 2))  # indices will be [[0, 1], [2, 1]], values will be [[6., 2.], [5., 4.]]
        # We need to create full indices like [[0, 0], [0, 1], [1, 2], [1, 1]]
        my_range = tf.expand_dims(tf.range(0, tf.shape(indices)[0]), 1)  # will be [[0], [1]]
        my_range_repeated = tf.tile(my_range, [1, int(self.topk / 2)])  # will be [[0, 0], [1, 1]]
        full_indices = tf.stack([my_range_repeated, indices], axis=2)  # change shapes to [N, k, 1] and [N, k, 1], to concatenate into [N, k, 2]
        full_indices = tf.reshape(full_indices, [-1, 2])
        P_reset = tf.sparse_to_dense(full_indices, tf.shape(self.sampling_doc_embed_aggregate), tf.reshape(values, [-1]), default_value=0., validate_indices=False)

        values2, indices2 = tf.nn.top_k(-N, self.topk - int(self.topk / 2))
        my_range = tf.expand_dims(tf.range(0, tf.shape(indices2)[0]), 1)
        my_range_repeated = tf.tile(my_range, [1, self.topk - int(self.topk / 2)])
        full_indices2 = tf.stack([my_range_repeated, indices2], axis=2)
        full_indices2 = tf.reshape(full_indices2, [-1, 2])
        N_reset = tf.sparse_to_dense(full_indices2, tf.shape(self.sampling_doc_embed_aggregate), tf.reshape(values2, [-1]), default_value=0., validate_indices=False)

        # 1)
        # res = P_reset - N_reset
        # tmp = 1 * batch_size * tf.reduce_sum(x - res, 1, keep_dims=True) / topk

        # P_reset = tf.sparse_to_dense(full_indices, tf.shape(x), tf.reshape(tf.add(values, tf.abs(tmp)), [-1]), default_value=0., validate_indices=False)
        # N_reset = tf.sparse_to_dense(full_indices2, tf.shape(x), tf.reshape(tf.add(values2, tf.abs(tmp)), [-1]), default_value=0., validate_indices=False)

        # 2)
        # factor = 0.
        # factor = 2. / topk
        P_tmp = self.alpha * tf.reduce_sum(P - P_reset, 1, keep_dims=True)  # 6.26
        N_tmp = self.alpha * tf.reduce_sum(-N - N_reset, 1, keep_dims=True)
        P_reset = tf.sparse_to_dense(full_indices, tf.shape(self.sampling_doc_embed_aggregate), tf.reshape(tf.add(values, P_tmp), [-1]), default_value=0., validate_indices=False)
        N_reset = tf.sparse_to_dense(full_indices2, tf.shape(self.sampling_doc_embed_aggregate), tf.reshape(tf.add(values2, N_tmp), [-1]), default_value=0., validate_indices=False)

        self.sparse_embed = P_reset - N_reset

        return self.sparse_embed

    def decoder(self):

        output_logits = tf.add(tf.matmul(self.sparse_embed, tf.transpose(self.weights['encoder_w'])), self.biases['decoder_b'])

        y_pred = output_logits
        y_true = tf.gather(self.doc, self.sampling_links[:, 1])
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y_true))

        return loss

    def add_contractive(self):

        if self.contractive != 0:
            derivative = tf.multiply(self.sparse_embed, 1 - self.sparse_embed)
            contractive_term = tf.reduce_sum(tf.matmul(tf.square(derivative), tf.expand_dims(tf.reduce_sum(tf.square(self.weights['encoder_w']), axis=0), 1)))
            return self.contractive * contractive_term
        else:
            return 0

    def construct_model(self):

        self.add_noise()
        self.encoder()
        self.neighbor_competition()
        self.sparse_encoding()
        loss = self.decoder()
        loss += self.add_contractive()

        return loss

    def train(self):

        loss = self.construct_model()
        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)

            num_minibatch = int(np.ceil(self.num_training_links / self.minibatch_size))
            t = time.time()
            one_epoch_loss = 0

            for epoch_index in range(1, self.num_epoch + 1):
                for minibatch_index in range(1, num_minibatch + 1):
                    _, one_epoch_loss = sess.run([optimizer, loss], feed_dict={self.sampling_links: self.data.minibatch_data[minibatch_index]['sampling_links'],
                                                                               self.doc: self.data.input_training,
                                                                               self.neighbor_ids: self.data.minibatch_data[minibatch_index]['neighbor_ids'],
                                                                               self.segment_ids: self.data.minibatch_data[minibatch_index]['segment_ids'],
                                                                               self.sm: self.sigma,
                                                                               })

                if epoch_index % 100 == 0 or epoch_index == 1:
                    print('******************************************************')
                    print('Time: %ds' % (time.time() - t), '\tEpoch: %d/%d' % (epoch_index, self.num_epoch), '\tLoss: %f' % one_epoch_loss)

                    doc_embed_training = sess.run(self.doc_embed, feed_dict={self.doc: self.data.input_training, self.sm: 0})
                    doc_embed_test = sess.run(self.doc_embed, feed_dict={self.doc: self.data.input_test, self.sm: 0})

                    if self.trans_induc == 'transductive':
                        classification_knn(self.trans_induc, vertices_embed=doc_embed_training, label=self.data.label_training)
                        clustering_kmeans(self.trans_induc, vertices_embed=doc_embed_training, label=self.data.label_training)
                    elif self.trans_induc == 'inductive':
                        classification_knn(self.trans_induc, X_train=doc_embed_training, X_test=doc_embed_test, Y_train=self.data.label_training, Y_test=self.data.label_test)
                        clustering_kmeans(self.trans_induc, X_train=doc_embed_training, X_test=doc_embed_test, Y_train=self.data.label_training, Y_test=self.data.label_test)

                    np.savetxt('./results/' + self.dataset_name + '_' + str(self.num_topics) + '_training.txt', doc_embed_training, delimiter='\t')
                    np.savetxt('./results/' + self.dataset_name + '_' + str(self.num_topics) + '_test.txt', doc_embed_test, delimiter='\t')

            print('Finish training! Training time:', time.time() - t)

            doc_embed_training = sess.run(self.doc_embed, feed_dict={self.doc: self.data.input_training, self.sm: 0})
            doc_embed_test = sess.run(self.doc_embed, feed_dict={self.doc: self.data.input_test, self.sm: 0})
            np.savetxt('./results/' + self.dataset_name + '_' + str(self.num_topics) + '_training.txt', doc_embed_training, delimiter='\t')
            np.savetxt('./results/' + self.dataset_name + '_' + str(self.num_topics) + '_test.txt', doc_embed_test, delimiter='\t')
            print('Finish saving embeddings!')