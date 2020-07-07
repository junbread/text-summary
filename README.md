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

**NOTE**: Now all requirements are included. No need to download.

* [khaiii](https://github.com/kakao/khaiii)
* [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/)

## Setup

**NOTE**: Automated Bash script is included. Now you just need to run `run_demo.sh` only.

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

Since datasets are too large to upload on Github, the files are uploaded on Google Drive. Download with links below:

* [Korean Dataset](https://drive.google.com/open?id=13zba2ti0qgTCEvHXQJO87NgDqe9eyT3f) (brought from BigKinds news data using [news-crawler](https://github.com/junbread/news-crawler))
* [English Dataset](https://drive.google.com/open?id=16MWCEySVq_39OhPrIYC6kDh0GcYlqmhA) (brought from CNN/Dailymail dataset, using [cnn-dailymail](https://github.com/abisee/cnn-dailymail))
* [Korean Pretrained Model](https://drive.google.com/open?id=14ksM6g6LojeY3ee1i9A_4t3W_VXvxfew)
* [English Pretrained Model](https://drive.google.com/open?id=1gyOL83VKaT3JMzJceEoL--xLk95EiaJ_) (brought from [here](https://github.com/abisee/pointer-generator#looking-for-pretrained-model))

### Extract data

Dataset location

```bash
project-root/data/
```

Model location

```bash
project-root/src/summarize/pgn/model/
```

Note that you have to extract content only. **Do not create subdirectory** under the location.

## How to run the demo

The demo page is made with Django framework. To run the demo, try this:

```bash
python src/demo/manage.py runserver
```
