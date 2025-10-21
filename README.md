[![PyPI version fury.io](https://badge.fury.io/py/bsparse.svg)](https://pypi.python.org/pypi/bsparse/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Worfklow](https://github.com/andrewyates/bsparse/workflows/pytest/badge.svg)](https://github.com/andrewyates/bsparse/actions)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
# bsparse
bsparse is a toolkit for creating and searching learned sparse representations

## Usage examples
```
# Recommended way to install requirements:
# (using pip only works too, but uv is much faster)
pipx install uv
# Create virtual environment
uv venv venv
# Activate
source venv/bin/activate
# Install requirements
uv pip install -r requirements.txt
```

```
# Request access to splade-v3: https://huggingface.co/naver/splade-v3
# Get your huggingface API token and then:
export HF_TOKEN="the token"

# load Python virtual environment
source venv/bin/activate

# optional: spot check output from a model
python -m bsparse.cli check --text "tesla net worth"

# create query representations:
python -m bsparse.cli encode --out nfcorpus-queries.jsonl \
  --dataset irds --type query --name beir/nfcorpus  --batch-size 64

# create doc representations:
python -m bsparse.cli encode --out nfcorpus-docs.jsonl \
  --dataset irds --type doc --name beir/nfcorpus  --batch-size 64

# search and evaluate without building an index:
python -m bsparse.cli memsearch --out nfcorpus.run --docs nfcorpus-docs.jsonl --queries nfcorpus-queries.jsonl --qrels beir/nfcorpus/test


# alternatively, you can build an index and search it

# 1) setup: compile ScaledJsonVectorCollection.java and add it to anserini-1.0.0-fatjar.jar
$ wget -c https://repo1.maven.org/maven2/io/anserini/anserini/1.0.0/anserini-1.0.0-fatjar.jar
$ cd java
$ javac -cp ../anserini-1.0.0-fatjar.jar io/anserini/collection/*.java
$ cp ../anserini-1.0.0-fatjar.jar ../anserini-1.0.0-fatjar-bsparse.jar
$ jar uf ../anserini-1.0.0-fatjar-bsparse.jar io/anserini/collection/*.class

# 2) build index
java -cp anserini-1.0.0-fatjar-AY.jar  io.anserini.index.IndexCollection \
  -generator DefaultLuceneDocumentGenerator -impact -pretokenized \
  -threads 16 -collection ScaledJsonVectorCollection \
  -input /path/to/encoded-text -index /path/to/encoded-text-index

# 3) search index
# Create sparse query representations in `$QUERY_VECTORS` and create an index in `$INDEX`, then:
python -m bsparse.cli search --index $INDEX --queries $QUERY_VECTORS --out test.run --topk 1000
```
