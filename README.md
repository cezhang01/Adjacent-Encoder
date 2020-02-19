# Adjacent-Encoder
This is the tensorflow implementation of the AAAI-2020 paper "Topic Modeling on Document Networks with Adjacent-Encoder" by Ce Zhang and Hady W. Lauw.
Adjacent-Encoder is a topic model that extracts topics for networked documents for document classification, clustering, link prediction, etc.
## Implementation Environment
- Python == 3.6
- Tensorflow == 1.9.0
- Numpy == 1.17.4
## Run
`python main.py`
### Parameter Setting
- -lr: learning rate, default=0.001
- -ne: number of epoch, default=1000
- -ti: transductive or inductive learning, default=inductive
- -dn: dataset name
- -nt: number of topics, default=64
- -x: Adjacent-Encoder or Adjacent-Encoder-X, default=0 (Adjacent-Encoder)
- -tr: training ratio, default=0.8
- -ms: minibatch size, default=128
- -sm: sigma used for Denoising Adjacent-Encoder(-X), default=0
- -c: contractive regularizer for Contractive Adjacent-Encoder(-X), default=0
- -sp: sparsity for K-Sparse Adjacent-Encoder(-X), default=0
- -k: number of nonzero hidden neurons of K-Sparse Adjacent-Encoder(-X), default=0.5 * num_topics
