import argparse
import collections
import json
import os
from abc import ABC, abstractmethod

import ir_datasets as irds
from datasets import load_dataset, load_from_disk


class Dataset(ABC, collections.abc.Iterable):
    @classmethod
    @abstractmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add dataset-specific arguments to the parser"""
        pass


class HgfDataset(Dataset):
    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument("--path", type=str, required=True, help="Dataset path")
        parser.add_argument("--name", type=str, default=None, help="Dataset name (default: %(default)s)")
        parser.add_argument("--split", type=str, default=None, help="Dataset split (default: %(default)s)")
        parser.add_argument("--id-field", type=str, default="id", help="Field containing doc/query ID (default: %(default)s)")
        parser.add_argument("--fields", type=str, nargs="*", default=["text"], help="List of text fields to include")
        parser.add_argument("--shard", type=int, default=0, help="Shard number (default: %(default)s)")
        parser.add_argument("--total-shards", type=int, default=1, help="Total number of shards (default: %(default)s)")

    def __init__(self, config):
        config.path = os.path.expanduser(config.path)

        if os.path.exists(config.path):
            print(f"loading dataset from local path: {config.path}")
            self.ds = load_from_disk(config.path, name=config.name, split=config.split).with_format("torch")
        else:
            print(f"loading dataset from huggingface datasets hub: {config.path}")
            self.ds = load_dataset(config.path, name=config.name, split=config.split, trust_remote_code=True).with_format("torch")

        self.id_field = config.id_field
        self.text_fields = config.fields
        self.get_text = lambda d: " ".join(d.get(field, "").strip() for field in self.text_fields).strip()

        self.shard = config.shard
        self.total_shards = config.total_shards

        assert self.total_shards >= 1
        assert self.shard >= 0 and self.shard < self.total_shards

        if self.total_shards == 1:
            assert self.shard == 0

        if self.total_shards > 1:
            print("sharding dataset")
            self.ds = self.ds.shard(num_shards=self.total_shards, index=self.shard)

    def __iter__(self):
        for d in self.ds:
            text = self.get_text(d)
            if text:
                yield (d[self.id_field], text)

    def __len__(self):
        return len(self.ds)


class TSVDataset(collections.abc.Iterable):
    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument("--fn", type=str, required=True, help="TSV file")
        parser.add_argument("--shard", type=int, default=0, help="Shard number (default: %(default)s)")
        parser.add_argument("--total-shards", type=int, default=1, help="Total number of shards (default: %(default)s)")

    def __init__(self, config):
        self.fn = os.path.expanduser(config.fn)
        self.shard = config.shard
        self.total_shards = config.total_shards

        assert self.total_shards >= 1
        assert self.shard >= 0 and self.shard < self.total_shards

        if self.total_shards == 1:
            assert self.shard == 0

        self._len = None

    def __iter__(self):
        with open(self.fn, "rt", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx % self.total_shards == self.shard:
                    fields = line.strip().split("\t")
                    if fields:
                        assert len(fields) == 2
                        yield fields

    def __len__(self):
        if not self._len:
            self._len = sum(1 for _ in self.__iter__())
        return self._len


class JSONLDataset(Dataset):
    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument("--fn", type=str, required=True, help="JSONL file")
        parser.add_argument("--id-field", type=str, default="id", help="Field containing doc/query ID (default: %(default)s)")
        parser.add_argument("--fields", type=str, nargs="*", default=["text"], help="List of text fields to include")
        parser.add_argument("--shard", type=int, default=0, help="Shard number (default: %(default)s)")
        parser.add_argument("--total-shards", type=int, default=1, help="Total number of shards (default: %(default)s)")

    def __init__(self, config):
        self.fn = os.path.expanduser(config.fn)
        self.id_field = config.id_field
        self.text_fields = config.fields
        self.shard = config.shard
        self.total_shards = config.total_shards

        assert self.total_shards >= 1
        assert self.shard >= 0 and self.shard < self.total_shards

        if self.total_shards == 1:
            assert self.shard == 0

        self.get_text = lambda d: " ".join(d.get(field, "").strip() for field in self.text_fields).strip()

        self._len = None

    def __iter__(self):
        if self.fn.endswith(".gz"):
            import gzip

            f = gzip.open(self.fn, "rt", encoding="utf-8")
        else:
            f = open(self.fn, "rt", encoding="utf-8")

        for idx, line in enumerate(f):
            if idx % self.total_shards == self.shard:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    print("JSONDecodeError:", line)
                    continue
                text = self.get_text(d)
                if text:
                    yield (d[self.id_field], text)

    def __len__(self):
        if not self._len:
            self._len = sum(1 for _ in self.__iter__())
        return self._len


class IRDSDataset(Dataset):
    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument("--name", type=str, default="beir/nfcorpus", help="IRDS dataset name (default: %(default)s)")
        parser.add_argument(
            "--type", type=str, default="doc", choices=["query", "doc"], help="Type of data (default: %(default)s)"
        )
        parser.add_argument("--fields", type=str, nargs="*", default=None, help="List of fields to include")
        parser.add_argument("--shard", type=int, default=0, help="Shard number (default: %(default)s)")
        parser.add_argument("--total-shards", type=int, default=1, help="Total number of shards (default: %(default)s)")

    def __init__(self, config):
        self.dataset = config.name
        self.type = config.type
        self.fields = config.fields
        self.shard = config.shard
        self.total_shards = config.total_shards

        assert self.total_shards >= 1
        assert self.shard >= 0 and self.shard < self.total_shards

        if self.total_shards == 1:
            assert self.shard == 0

        self.ds = irds.load(self.dataset)

        if self.fields:
            self.get_text = lambda x: " ".join(getattr(x, field).strip() for field in self.fields)
        else:
            self.get_text = lambda x: x.default_text().strip()

    def __iter__(self):
        if self.type == "query":
            yield from self._query_iter()
        elif self.type == "doc":
            yield from self._doc_iter()
        else:
            raise ValueError(f"unknown type: {self.type}")

    def __len__(self):
        if self.type == "query":
            total = self.ds.queries_count()
        elif self.type == "doc":
            total = self.ds.docs_count()
        else:
            raise ValueError(f"unknown type: {self.type}")

        count = total // self.total_shards
        remainder = total % self.total_shards
        if self.shard < remainder:
            count += 1
        return count

    def _query_iter(self):
        for idx, query in enumerate(self.ds.queries_iter()):
            if idx % self.total_shards == self.shard:
                yield (query.query_id, self.get_text(query))

    def _doc_iter(self):
        for idx, doc in enumerate(self.ds.docs_iter()):
            if idx % self.total_shards == self.shard:
                yield (doc.doc_id, self.get_text(doc))
