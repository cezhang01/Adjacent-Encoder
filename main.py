import argparse
import numpy as np
import tensorflow as tf

from data_preparation import Data
from adjacent_encoder import adjacent_encoder


def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-ne', '--num_epoch', type=int, default=1000)
    parser.add_argument('-ti', '--trans_induc', type=str, default='inductive', help='transductive or inductive, transductive means we input all documents and links for unsupervised training, inductive means we split 80% for training, 20% for test')
    parser.add_argument('-dn', '--dataset_name', type=str, default='ds', help='ds or ha or ml or pl')
    parser.add_argument('-nt', '--num_topics', type=int, default=64)
    parser.add_argument('-x', '--x', type=int, default=0, help='0 == Adjacent-Encoder, 1 == Adjacent-Encoder-X')
    parser.add_argument('-tr', '--training_ratio', type=float, default=0.8, help='This program will automatically split 10% among training set for validation')
    parser.add_argument('-ms', '--minibatch_size', type=int, default=128)

    parser.add_argument('-sm', '--sigma', type=float, default=0, help='gaussian std.dev. used for Denoising Adjacent-Encoder(-X), if do not want to use denoising variant, set this value = 0')
    parser.add_argument('-c', '--contractive', type=float, default=0, help='used for Contractive Adjacent-Encoder(-X), best performance = 1e-11, if do not want to use contractive variant, set this value = 0')
    parser.add_argument('-sp', '--sparsity', type=int, default=0, help='0 == no k-sparse, 1 == K-Sparse Adjacent-Encoder(-X)')
    parser.add_argument('-k', '--topk', type=int, default=32, help='number of nonzero hidden neurons used for K-Sparse Adjacent-Encoder(-X), best performance = 0.5 * num_hidden, if do not use k-sparse variant, set above argument -sp to 0')

    parser.add_argument('-rs', '--random_seed', type=int, default=950)

    return parser.parse_args()


def main(args):

    if args.random_seed:
        tf.set_random_seed(args.random_seed)
        np.random.seed(args.random_seed)
    print('Preparing data...')
    data = Data(args)
    print('Initializing model...')
    model = adjacent_encoder(args, data)
    print('Start training...')
    model.train()


if __name__ == '__main__':
    main(parse_arguments())