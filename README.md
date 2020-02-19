# Adjacent-Encoder
This is the tensorflow implementation of the AAAI-2020 paper "Topic Modeling on Document Networks with Adjacent-Encoder" by Ce Zhang and Hady W. Lauw.

Adjacent-Encoder is a topic model that extracts topics for networked documents for document classification, clustering, link prediction, etc.

![](file:///C:/Users/cezhang.2018/Dropbox/_shared/Papers/aaai20_adjacent_encoder/camera_ready/AAAI-ZhangC.2977/model_comparison.pdf)
## Implementation Environment
- Python == 3.6
- Tensorflow == 1.9.0
- Numpy == 1.17.4

## Run
`python main.py`

### Parameter Setting
- -lr: learning rate, default = 0.001
- -ne: number of epoch, default = 1000
- -ti: transductive or inductive learning, transductive means we input all documents and links for unsupervised training, inductive means we split 80% for training, 20% for test, default = inductive
- -dn: dataset name
- -nt: number of topics, default = 64
- -x: 0 == Adjacent-Encoder, 1 == Adjacent-Encoder-X, default = 0 (Adjacent-Encoder)
- -tr: training ratio, this program will automatically split 10% among training set for validation, default = 0.8
- -ms: minibatch size, default = 128
- -sm: gaussian std.dev. (sigma) used for Denoising Adjacent-Encoder(-X), if do not want to use denoising variant, set this value to 0, default = 0
- -c: contractive regularizer for Contractive Adjacent-Encoder(-X), best performance = 1e-11, if do not want to use contractive variant, set this value to 0, default = 0
- -sp: 0 == no k-sparse, 1 == K-Sparse Adjacent-Encoder(-X), default = 0
- -k: number of nonzero hidden neurons of K-Sparse Adjacent-Encoder(-X), if do not use k-sparse variant, set above argument -sp to 0, default = 0.5 * num_topics

## Data
We extracted four independent datasets (DS, HA, ML, and PL) from source Cora (http://people.cs.umass.edu/~mccallum/data/cora-classify.tar.gz).

In `./cora` file we release these datasets, each of which contains adjacency matrix, content, label, and vocabulary.

- adjacency matrix (NxN): a 0-1 symmetric matrix (A^T==A), and its diagonal elements are supposed to be 1.
- content (Nx|V|): each row is a Bag-of-Words representation of the corresponding document, and each column is a word in the vocabulary. Documents are represented by the word count.
- label (Nx1): label or category of each document.
- vocabulary (|V|x1): words.

## Reference
If you use our paper, including code and data, please cite

```
@inproceedings{adjenc,
    title={Topic Modeling on Document Networks with Adjacent-Encoder},
    author={Zhang, Ce and Lauw, Hady W},
    booktitle={Thirty-fourth AAAI conference on artificial intelligence},
    year={2020}
}
```
