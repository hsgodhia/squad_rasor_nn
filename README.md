# Learning Recurrent Span Representations for Extractive Question Answering
[https://arxiv.org/abs/1611.01436](https://arxiv.org/abs/1611.01436)

#### Initial setup

```bash
$ python setup.py
```
This will download GloVe word embeddings and tokenize raw training / development data.<br />
(download will be skipped if [zipped GloVe file](http://nlp.stanford.edu/data/glove.840B.300d.zip) is manually placed in `data` directory).

#### Training

```bash
$ python train_main.py
```

#### Making predictions

```bash
$ python predict_main.py
```

---

Tested in the following environment:

* Ubuntu 16.04
* Python 2.7.13
* NVIDIA CUDA 7.5 and cuDNN 5.1
* Pytorch
* Oracle JDK 8