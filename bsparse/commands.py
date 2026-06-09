import json
from abc import ABC, abstractmethod
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import ir_datasets as irds
from tqdm import tqdm
from trecrun import TRECRun

from bsparse import load_dict, save_dict
from bsparse.anserini import Anserini
from bsparse.seismic import Seismic
from bsparse.utils import psgid_to_docid


class Command(ABC):
    needs_model = False
    needs_dataset = False

    @classmethod
    @abstractmethod
    def add_arguments(cls, parser) -> None:
        """Add Command-specific arguments to the parser"""
        pass


class Check(Command):
    needs_model = True
    needs_dataset = False

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument("--text", type=str, required=True, help="Text to encode")

    def __init__(self, config, model):
        self.cfg = config
        self.model = model

    def run(self):
        dataset = [("text", self.cfg.text)]
        ids, reps = self.model.encode(dataset)
        print("encoded text:", self.cfg.text)
        print(json.dumps(reps, indent=2))


class Encode(Command):
    needs_model = True
    needs_dataset = True

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument("--out", type=Path, required=True, help="Output file path")

    def __init__(self, config, model, dataset):
        self.cfg = config
        self.model = model
        self.dataset = dataset

    def run(self):
        self.cfg.out.parent.mkdir(parents=True, exist_ok=True)

        ids, reps = self.model.encode(self.dataset)
        assert len(ids) == len(reps)
        save_dict(reps, ids, self.cfg.out, term2id=None)


def score_shard(docs_fn, queries_fn, topk=1000, aggregate="maxscore"):
    queries = load_dict(queries_fn)
    docs = load_dict(docs_fn)

    if aggregate == "maxrep":
        d = {}
        for psgid, psgrep in zip(docs.ids, docs.weights):
            docid = psgid_to_docid(psgid)
            d.setdefault(docid, {})
            d[docid] = {term: max(psgrep.get(term, 0), d[docid].get(term, 0)) for term in d[docid] | psgrep}

        docpairs = d.items()
    else:
        docpairs = zip(docs.ids, docs.weights)

    queries.ids = [psgid_to_docid(qid) for qid in queries.ids]
    scores = {qid: {} for qid in queries.ids}
    for docid, docrep in docpairs:
        for qid, qrep in zip(queries.ids, queries.weights):
            score = sum(qweight * docrep[qterm] for qterm, qweight in qrep.items() if qterm in docrep)
            if score:
                scores[qid][docid] = score

    topk_scores = TRECRun(scores).aggregate_docids(psgid_to_docid).topk(topk).results
    return topk_scores


class MemSearch(Command):
    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument("--docs", type=Path, nargs="+", required=True, help="List of document file paths")
        parser.add_argument("--queries", type=Path, required=True, help="Query file path")
        parser.add_argument("--out", type=Path, required=True, help="Output file path")
        parser.add_argument("--pool", type=int, default=20, help="Multiprocessing pool size (default: %(default)s)")
        parser.add_argument("--topk", type=int, default=1000, help="Top K results to return (default: %(default)s)")
        parser.add_argument("--qrels", type=str, default=None, help="Relevance judgments dataset (default: %(default)s)")
        parser.add_argument(
            "--aggregate",
            type=str,
            default="maxscore",
            choices=["maxscore", "maxrep"],
            help="Aggregation method (default: %(default)s)",
        )

    def __init__(self, config):
        self.cfg = config

    def run(self):
        if self.cfg.out.is_dir():
            raise ValueError(f"--out is a directory: {self.cfg.out}")

        with Pool(self.cfg.pool) as p:
            score_f = partial(score_shard, queries_fn=self.cfg.queries, topk=self.cfg.topk, aggregate=self.cfg.aggregate)
            shard_scores = list(
                tqdm(
                    p.imap(score_f, self.cfg.docs),
                    total=len(self.cfg.docs),
                    desc="memsearch: ranking",
                    leave=False,
                )
            )

        combined_scores = {}
        neginf = float("-inf")
        for shard_score in tqdm(shard_scores, desc="memsearch: merging", leave=False):
            for qid in shard_score:
                combined_scores.setdefault(qid, {})
                for docid, score in shard_score[qid].items():
                    combined_scores[qid][docid] = max(score, combined_scores[qid].get(docid, neginf))

        run = TRECRun(combined_scores).aggregate_docids(psgid_to_docid).topk(self.cfg.topk)

        print(f"saving run to: {self.cfg.out}")
        run.write_trec_run(self.cfg.out)

        if self.cfg.qrels:
            print(f"evaluating with qrels: {self.cfg.qrels}")
            all_qrels = {}
            for qr in irds.load(self.cfg.qrels).qrels_iter():
                all_qrels.setdefault(qr.query_id, {})[qr.doc_id] = qr.relevance

            metrics = run.evaluate(all_qrels)
            avg = {metric: vals["mean"] for metric, vals in metrics.items()}
            print(json.dumps(avg, indent=4, sort_keys=True))

        return run


class Index(Command):
    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument(
            "--input",
            type=Path,
            nargs="+",
            required=True,
            help="One or more encoded JSONL doc files (plain or .gz), or directories of such files",
        )
        parser.add_argument("--index", type=Path, required=True, help="Output index path")
        parser.add_argument(
            "--backend", type=str, default="seismic", choices=["seismic"], help="Indexing backend (default: %(default)s)"
        )
        # Seismic index-building hyperparameters (these affect the index, so they are kwargs, not env vars).
        # Defaults match the recommended config in the Seismic guidelines.
        parser.add_argument(
            "--n-postings", type=int, default=3000, help="[Seismic only] avg postings per list (default: %(default)s)"
        )
        parser.add_argument(
            "--centroid-fraction",
            type=float,
            default=0.2,
            help="[Seismic only] centroids per list as a fraction (default: %(default)s)",
        )
        parser.add_argument(
            "--summary-energy",
            type=float,
            default=0.5,
            help="[Seismic only] fraction of summary L1 norm to keep (default: %(default)s)",
        )
        parser.add_argument(
            "--min-cluster-size", type=int, default=2, help="[Seismic only] minimum cluster size (default: %(default)s)"
        )
        parser.add_argument(
            "--max-fraction",
            type=float,
            default=6,
            help="[Seismic only] max summary block size as a fraction (default: %(default)s)",
        )
        parser.add_argument(
            "--nknn", type=int, default=0, help="[Seismic only] kNN graph size; 0 disables it (default: %(default)s)"
        )
        parser.add_argument(
            "--batched-indexing",
            type=int,
            default=100000,
            help="[Seismic only] docs per indexing batch (default: %(default)s)",
        )
        parser.add_argument(
            "--variant",
            type=str,
            default="standard",
            choices=["standard", "large_vocab"],
            help="[Seismic only] index variant; use large_vocab for >65k tokens (default: %(default)s)",
        )
        parser.add_argument(
            "--build-method",
            type=str,
            default="dataset",
            choices=["dataset", "file"],
            help="[Seismic only] feed docs via the in-memory dataset API, or via a temporary "
            "uncompressed JSONL file as a fallback (default: %(default)s)",
        )

    def __init__(self, config):
        self.cfg = config

    def run(self):
        if self.cfg.backend == "seismic":
            Seismic.build(
                self.cfg.input,
                self.cfg.index,
                n_postings=self.cfg.n_postings,
                centroid_fraction=self.cfg.centroid_fraction,
                summary_energy=self.cfg.summary_energy,
                min_cluster_size=self.cfg.min_cluster_size,
                max_fraction=self.cfg.max_fraction,
                nknn=self.cfg.nknn,
                batched_indexing=self.cfg.batched_indexing,
                variant=self.cfg.variant,
                method=self.cfg.build_method,
            )
        else:
            raise ValueError(f"unknown indexing backend: {self.cfg.backend}")


class Search(Command):
    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument("--index", type=Path, required=True, help="Index path")
        parser.add_argument("--queries", type=Path, required=True, help="Query file path")
        parser.add_argument("--out", type=Path, required=True, help="Output file path")
        parser.add_argument(
            "--backend",
            type=str,
            default="anserini",
            choices=["anserini", "seismic"],
            help="Search backend (default: %(default)s)",
        )
        parser.add_argument("--topk", type=int, default=1000, help="Top K results to return (default: %(default)s)")
        parser.add_argument("--qrels", type=str, default=None, help="Relevance judgments dataset (default: %(default)s)")
        # Backend-specific args default to None so we can tell whether the user set them: unset args fall
        # back to the backend's own default, and passing an arg for a different backend is rejected (see run()).
        # anserini-specific
        parser.add_argument("--scale", type=int, default=None, help="[Anserini only] impact scaling factor (default: 50)")
        # seismic-specific
        parser.add_argument("--query-cut", type=int, default=None, help="[Seismic only] query_cut (default: 10)")
        parser.add_argument("--heap-factor", type=float, default=None, help="[Seismic only] heap_factor (default: 0.8)")

    # maps each backend to the args that only apply to it
    BACKEND_ARGS = {"anserini": ["scale"], "seismic": ["query_cut", "heap_factor"]}

    def __init__(self, config):
        self.cfg = config

    def _check_backend_args(self):
        """Reject args that only apply to a backend other than the one selected."""
        for backend, names in self.BACKEND_ARGS.items():
            if backend == self.cfg.backend:
                continue
            misused = [f"--{name.replace('_', '-')}" for name in names if getattr(self.cfg, name) is not None]
            if misused:
                raise ValueError(f"{', '.join(misused)} only valid with the {backend} backend, not '{self.cfg.backend}'")

    def run(self):
        self._check_backend_args()

        if self.cfg.out.is_dir():
            raise ValueError(f"--out is a directory: {self.cfg.out}")

        queries = load_dict(self.cfg.queries)
        queries.ids = [psgid_to_docid(qid) for qid in queries.ids]
        vectors = [{"vector": rep} for rep in queries.weights]

        if self.cfg.backend == "anserini":
            retriever = Anserini(self.cfg.index.as_posix())
            kwargs = {} if self.cfg.scale is None else {"scale": self.cfg.scale}
            results = retriever.query_from_vectors(vectors, k=self.cfg.topk, **kwargs)
        elif self.cfg.backend == "seismic":
            retriever = Seismic(self.cfg.index.as_posix())
            kwargs = {}
            if self.cfg.query_cut is not None:
                kwargs["query_cut"] = self.cfg.query_cut
            if self.cfg.heap_factor is not None:
                kwargs["heap_factor"] = self.cfg.heap_factor
            results = retriever.query_from_vectors(vectors, k=self.cfg.topk, **kwargs)
        else:
            raise ValueError(f"unknown search backend: {self.cfg.backend}")

        run = TRECRun(dict(zip(queries.ids, results))).aggregate_docids(psgid_to_docid).topk(self.cfg.topk)
        print(f"saving run to: {self.cfg.out}")
        run.write_trec_run(self.cfg.out)

        if self.cfg.qrels:
            print(f"evaluating with qrels: {self.cfg.qrels}")
            all_qrels = {}
            for qr in irds.load(self.cfg.qrels).qrels_iter():
                all_qrels.setdefault(qr.query_id, {})[qr.doc_id] = qr.relevance

            metrics = run.evaluate(all_qrels)
            avg = {metric: vals["mean"] for metric, vals in metrics.items()}
            print(json.dumps(avg, indent=4, sort_keys=True))

        return run
