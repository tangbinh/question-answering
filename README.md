### Overview
This is a curated PyTorch implementation of the ACL 2017 paper [Reading Wikipedia to Answer Open-Domain Questions](https://arxiv.org/pdf/1704.00051.pdf) (DrQA). Compared to the code in [ facebookresearch/DrQA](https://github.com/facebookresearch/DrQA/), this implementation doesn't have the document retriever and is only for purpose of training and evaluating on [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/), making it cleaner and much more light-weighted. Although there are quite a few differences in detail, this implementation follows closely the official code and achieves comparable results (F1 on the dev set being 77.6%).

### Installation
The code was written for Python 3.6 or higher, and it has been tested with [PyTorch](http://pytorch.org/) 0.4.1. Other dependencies are listed in [requirements.txt](https://github.com/tangbinh/question-answering/blob/master/requirements.txt). Training is only available with GPU. To get started, try to clone the repository

```bash
git clone https://github.com/tangbinh/question-answering
cd question-answering
pip install -r requirements.txt
```
There are two tokenizers included in this implementation, [CoreNLPTokenizer](https://stanfordnlp.github.io/CoreNLP/) and [SpacyTokenizer](https://spacy.io). Although spaCy is faster, its tokenizer actually results in more questions having no answers when mapped from texts to tokens. The default setting therefore requires CoreNLPTokenizer, so please download the Stanford CoreNLP jars and include them in your CLASSPATH by following these commands:

```bash
wget "http://nlp.stanford.edu/software/stanford-corenlp-full-2017-06-09.zip"
unzip "stanford-corenlp-full-2017-06-09.zip"
rm "stanford-corenlp-full-2017-06-09.zip"
mv "stanford-corenlp-full-2017-06-09" "corenlp"
export CLASSPATH=corenlp/*:$CLASSPATH
```

### Preprocessing
If you haven't downloaded SQuAD or GloVe, it might be easier for you to just run
```bash
bash download.sh
```
Then, the following command helps tokenize the downloaded datasets, extract features such as part-of-speech tagging, and build a dictionary using all CPUs available in your machine:
```bash
python preprocess.py --data data/squad --embed-path wordvec/glove/glove.840B.300d.txt --restrict-vocab
```

### Training
To get started with training a model on SQuAD, you might find the following command helpful:
```bash
python train.py --embed-path wordvec/glove/glove.840B.300d.txt --checkpoint-dir checkpoints/drqa --log-file logs/drqa.log
```

### Prediction
When the training is done, you can make predictions and run the official evaluation script:
```bash
python predict.py --input data/dev.json --output predictions/dev-pred.json --feature-dict data/feature_dict.json --checkpoint checkpoints/drqa/checkpoint_best.pt
python evaluate.py data/dev-v1.1.json predictions/dev-pred.json
```
### Interactive
As in the official implementation, it's possible to have an interactive environment for evaluation:
```bash
python interactive.py --checkpoint checkpoints/drqa/checkpoint_best.pt
```
```
>>> context = "Mary had a little lamb, whose fleece was white as snow. And everywhere that Mary went the lamb was sure to go."
>>> question = "What color is Mary's lamb?"
>>> answer(context, question)
+------+-------+--------------------+
| Rank |  Span |       Score        |
+------+-------+--------------------+
|  1   | white | 0.8701031804084778 |
+------+-------+--------------------+
Time: 0.2440
```