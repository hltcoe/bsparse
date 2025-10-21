import os
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass

from trecrun import TRECRun

from bsparse.models import Model
from bsparse.utils import psgid_to_docid


ANSERINI_JAR = os.environ.get("ANSERINI_JAR", "anserini-1.0.0-fatjar-bsparse.jar")
JAVA_ARGS = os.environ.get("ANSERINI_JAVA_ARGS", "-Xms4g,-Xmx16g").split(",")
THREADS = os.environ.get("ANSERINI_THREADS", os.cpu_count())


def sparse_rep_to_text(rep_dict, scale: int):
    return " ".join(Counter({term: int(scale * weight) for term, weight in rep_dict.items()}).elements())


class Anserini:
    def __init__(self, index_path: str):
        self.index_path = index_path

    def query_from_raw_text(self, queries: list[str], model: Model, k: int = 1000, scale: int = 50):
        dataset = [(str(idx), query) for idx, query in enumerate(queries)]
        ids, reps = model.encode(dataset)
        vectors = [{"vector": rep} for rep in reps]
        return self.query_from_vectors(vectors, k=k, scale=scale)

    def query_from_vectors(self, queries: list[dict], k: int = 1000, scale: int = 50):
        text_queries = [sparse_rep_to_text(d["vector"], scale=scale) for d in queries]
        return self._query_from_preprocessed_text(text_queries, k=k)

    def _query_from_preprocessed_text(self, queries: list[str], k: int = 1000):
        queries = dict(enumerate(queries))

        # these imports (slowly) load anserini's fatjar
        _init_anserini()
        from jnius import autoclass

        SearchCollection = autoclass("io.anserini.search.SearchCollection")

        tmp_path = os.environ.get("TMPDIR", "/tmp")
        os.makedirs(tmp_path, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=tmp_path + "/") as tmpdir:
            queryfn = os.path.join(tmpdir, "queries.tsv")
            runfn = os.path.join(tmpdir, "run.txt")

            with open(queryfn, "wt", encoding="utf-8") as queryf:
                for qid, text in queries.items():
                    print(f"{qid}\t{text}", file=queryf)

            args = [
                "-topicReader",
                "TsvString",
                "-index",
                self.index_path,
                "-topics",
                queryfn,
                "-output",
                runfn,
                "-threads",
                str(THREADS),
                "-hits",
                str(k),
                "-pretokenized",
                "-impact",
            ]

            SearchCollection.main(args)
            run = TRECRun(runfn).aggregate_docids(psgid_to_docid)

        results = [run[str(qid)] for qid, _ in enumerate(queries)]
        return results


def _init_anserini():
    import jnius_config

    if not jnius_config.vm_running:
        if not os.path.exists(ANSERINI_JAR):
            print(f"WARNING: missing Anserini JAR file: {ANSERINI_JAR}", file=sys.stderr)
            print("          Set the ANSERINI_JAR environment variable to point to a valid Anserini 1.0 jar", file=sys.stderr)
        jnius_config.add_options(*JAVA_ARGS)
        jnius_config.add_classpath(ANSERINI_JAR)


if __name__ == "__main__":
    # TODO clean this up
    _, query = sys.argv
    index = "/exp/ayates/scale25/neuclir/index/multilingual_lsr_60000"
    anserini = Anserini(index)

    @dataclass
    class Args:
        ckpt: str = "/exp/ayates/scale25/multilsr-checkpoints/xlmr_60000/"
        max_length: int = 512
        stride: int = 510
        batch_size: int = 8
        max_terms: int = None

    from bsparse.models import MultiLSRModel

    config = Args()
    model = MultiLSRModel(config)

    results = anserini.query_from_raw_text([query], model=model)
    # results = anserini._query_from_preprocessed_text([query])
    print(results)
