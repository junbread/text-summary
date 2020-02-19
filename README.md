# text-summary

Generates text summaries from given inputs through several methods. supports Korean only, for now.

Currently implemented methods:

* Pointer-Generator Network (most codes are from [this project](https://github.com/abisee/pointer-generator) with some modification.)
* TextRank (by `textrankr` library)
* ~~Transformer~~

This is a sub-project of [skku-coop-project](https://github.com/JunBread/skku-coop-project) backed by SK Planet.

## Requirements

### Python packages

```plaintext
hanja
textrankr
django
tensorflow >= 2.0.0
koalanlp
```

### Other requirements

* [khaiii](https://github.com/kakao/khaiii)
* [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/)

## Setup

Before running demo page server, you should export your working directory as `PYTHONPATH`. Try with this:

```bash
export PYTHONPATH=$PYTHONPATH:path/to/project
```

Also this project uses Stanford CoreNLP library to regularize input sentences. We assume you already installed Java runtime & downloaded CoreNLP library.

Like `PYTHONPATH`, to run the server you need to specify CoreNLP jar file into `CLASSPATH`. Try with this:

```bash
export CLASSPATH=$CLASSPATH:path/to/corenlp/stanford-corenlp-(version number).jar
```

## Download dataset (& pretrained model)

Not yed prepared (I will add as soon as possible)

## How to run the demo

The demo page is made with Django framework. To run the demo, try this:

```bash
python src/demo/manage.py runserver
```
