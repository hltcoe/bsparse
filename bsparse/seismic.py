import gzip
import json
import os
import sys
import tempfile
import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm


if TYPE_CHECKING:
    # only used as a type hint; importing bsparse.models would pull in torch/transformers
    from bsparse.models import Model


THREADS = int(os.environ.get("SEISMIC_THREADS", os.cpu_count()))

# Index-building hyperparameter defaults (override per-call via build() kwargs).
# These match the recommended configuration in the Seismic guidelines:
#   https://github.com/TusKANNy/seismic/blob/main/docs/Guidelines.md
#   build(n_postings=3000, centroid_fraction=0.2, min_cluster_size=2, summary_energy=0.5, max_fraction=6)
#   (reasonable centroid_fraction in {0.05, 0.1, 0.2}; summary_energy in [0.4, 0.6])
DEFAULT_N_POSTINGS = 3000
DEFAULT_CENTROID_FRACTION = 0.2
DEFAULT_SUMMARY_ENERGY = 0.5
DEFAULT_MIN_CLUSTER_SIZE = 2
DEFAULT_MAX_FRACTION = 6
# nknn=0 builds no kNN graph (not needed for basic indexing); batched_indexing only affects
# build speed/memory, not the resulting index
DEFAULT_NKNN = 0
DEFAULT_BATCHED_INDEXING = 100_000

# Query-time hyperparameter defaults.
#   query_cut ~ 10 generally yields high effectiveness; heap_factor in [0.7, 1].
DEFAULT_QUERY_CUT = 10
DEFAULT_HEAP_FACTOR = 0.8

# seismic's save(path) writes to path + this suffix, but load() expects the full on-disk filename
INDEX_SUFFIX = ".index.seismic"


def _resolve_index_path(path: str) -> str:
    """Resolve an index path to the on-disk filename, accepting either the exact saved file or the
    prefix that was passed to build() (to which seismic's save() appends INDEX_SUFFIX)."""
    if os.path.isfile(path):
        return path
    if os.path.isfile(path + INDEX_SUFFIX):
        return path + INDEX_SUFFIX
    raise FileNotFoundError(f"no Seismic index found at: {path} (also tried: {path + INDEX_SUFFIX})")


def _import_seismic():
    try:
        import seismic
    except ImportError:
        print("ERROR: the 'seismic' package is required for Seismic indexing/search.", file=sys.stderr)
        print("       Install it with: pip install pyseismic-lsr", file=sys.stderr)
        print(
            '       (for best performance: RUSTFLAGS="-C target-cpu=native" pip install --no-binary :all: pyseismic-lsr)',
            file=sys.stderr,
        )
        raise

    return seismic


# set once the truncation warning has fired, so we don't re-scan every document of a large build
_truncation_warned = False


def _to_seismic_strings(strings: list, string_type) -> np.ndarray:
    """Convert tokens to seismic's fixed-width unicode dtype, warning if any would be truncated.

    numpy silently truncates strings longer than the dtype's width (e.g. 30 chars for "U30").
    Truncated tokens can collide with each other or fail to match tokens that were stored without
    truncation (e.g. by the file-based build, which bypasses numpy), so this is worth a warning.
    """
    global _truncation_warned
    if not _truncation_warned:
        width = np.dtype(string_type).itemsize // np.dtype("U1").itemsize
        too_long = next((s for s in strings if len(s) > width), None)
        if too_long is not None:
            _truncation_warned = True
            warnings.warn(
                f"token longer than seismic's {width}-char string dtype will be truncated: {too_long!r} "
                "(truncated tokens may collide or fail to match; further warnings suppressed)"
            )
    return np.array(strings, dtype=string_type)


JSONL_SUFFIXES = (".jsonl", ".jsonl.gz")


def _expand_inputs(inputs) -> list[str]:
    """Normalize build inputs into a concrete list of JSONL files, expanding directories.

    `inputs` may be a single path or a list of paths; each entry may be a .jsonl/.jsonl.gz file or a
    directory (expanded to the .jsonl/.jsonl.gz files it directly contains, sorted). This is
    experiment code, so we fail loudly on paths that don't exist or directories with no matches
    rather than silently skipping them.
    """
    if isinstance(inputs, (str, os.PathLike)):
        inputs = [inputs]
    inputs = [str(p) for p in inputs]
    if not inputs:
        raise ValueError("no input files provided to Seismic.build")

    expanded = []
    for path in inputs:
        if os.path.isdir(path):
            matches = sorted(os.path.join(path, name) for name in os.listdir(path) if name.endswith(JSONL_SUFFIXES))
            if not matches:
                raise FileNotFoundError(f"no .jsonl/.jsonl.gz files found in directory: {path}")
            expanded.extend(matches)
        elif os.path.isfile(path):
            expanded.append(path)
        else:
            raise FileNotFoundError(f"Seismic.build input is not a readable file or directory: {path}")

    return expanded


def _iter_docs(inputs: list[str]):
    """Yield (doc_id, vector) pairs from encoded JSONL files (plain or .gz), failing loudly."""
    for fn in inputs:
        opener = gzip.open if fn.endswith(".gz") else open
        with opener(fn, "rt", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                try:
                    d = json.loads(line)
                    doc_id, vector = d["id"], d["vector"]
                except (json.JSONDecodeError, KeyError) as e:
                    raise ValueError(f"invalid document on line {line_no} of {fn}: {e}") from e

                yield str(doc_id), vector


@contextmanager
def _seismic_input(inputs: list[str]):
    """Yield a single uncompressed JSONL path that Seismic can build from.

    Seismic's build() only accepts one uncompressed JSONL file. A single plain-JSONL input is used
    as-is; otherwise (multiple files and/or .gz) the inputs are streamed line-by-line into a
    temporary uncompressed file (decompressing .gz on the way) that is deleted afterwards.
    """
    if len(inputs) == 1 and not inputs[0].endswith(".gz"):
        yield inputs[0]
        return

    tmp_dir = os.environ.get("TMPDIR", "/tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=tmp_dir + "/") as tmpdir:
        combined = os.path.join(tmpdir, "combined.jsonl")
        with open(combined, "wt", encoding="utf-8") as out:
            for fn in inputs:
                opener = gzip.open if fn.endswith(".gz") else open
                with opener(fn, "rt", encoding="utf-8") as f:
                    for line in f:
                        # guarantee a newline between concatenated files so we never merge two
                        # JSON objects onto one line
                        out.write(line if line.endswith("\n") else line + "\n")
        yield combined


class Seismic:
    def __init__(self, index_path: str):
        self.index_path = index_path
        self._index = None

    @property
    def index(self):
        # the index is loaded lazily so that constructing a Seismic object (and importing this
        # module) does not require the native 'seismic' package to be installed
        if self._index is None:
            self._index = _import_seismic().SeismicIndex.load(_resolve_index_path(self.index_path))
        return self._index

    def query_from_raw_text(
        self,
        queries: list[str],
        model: "Model",
        k: int = 1000,
        query_cut: int = DEFAULT_QUERY_CUT,
        heap_factor: float = DEFAULT_HEAP_FACTOR,
    ):
        dataset = [(str(idx), query) for idx, query in enumerate(queries)]
        ids, reps = model.encode(dataset)
        vectors = [{"vector": rep} for rep in reps]
        return self.query_from_vectors(vectors, k=k, query_cut=query_cut, heap_factor=heap_factor)

    def query_from_vectors(
        self,
        queries: list[dict],
        k: int = 1000,
        query_cut: int = DEFAULT_QUERY_CUT,
        heap_factor: float = DEFAULT_HEAP_FACTOR,
    ):
        string_type = _import_seismic().get_seismic_string()

        query_ids = [str(i) for i, _ in enumerate(queries)]
        query_components = [_to_seismic_strings(list(q["vector"].keys()), string_type) for q in queries]
        query_values = [np.array(list(q["vector"].values()), dtype=np.float32) for q in queries]

        results = self.index.batch_search(
            queries_ids=np.array(query_ids, dtype=string_type),
            query_components=query_components,
            query_values=query_values,
            k=k,
            query_cut=query_cut,
            heap_factor=heap_factor,
            num_threads=THREADS,
        )

        # batch_search returns one list per query, each a list of (query_id, score, doc_id) tuples.
        # we map by the returned query_id (rather than position) and return results aligned with the
        # input queries, matching the {docid: score} shape produced by Anserini.query_from_vectors
        scores_by_qid = {qid: {} for qid in query_ids}
        for query_results in results:
            for query_id, score, doc_id in query_results:
                scores_by_qid[query_id][doc_id] = score

        return [scores_by_qid[qid] for qid in query_ids]

    @staticmethod
    def build(
        inputs,
        index_path,
        n_postings: int = DEFAULT_N_POSTINGS,
        centroid_fraction: float = DEFAULT_CENTROID_FRACTION,
        summary_energy: float = DEFAULT_SUMMARY_ENERGY,
        min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
        max_fraction: float = DEFAULT_MAX_FRACTION,
        nknn: int = DEFAULT_NKNN,
        batched_indexing: int = DEFAULT_BATCHED_INDEXING,
        variant: str = "standard",
        method: str = "dataset",
    ):
        """Build a Seismic index from one or more encoded JSONL files and save it to index_path.

        Inputs are in the same format produced by `bsparse encode`: one JSON object per line with an
        "id" and a "vector" (a {token: weight} dict). Unlike Anserini, Seismic indexes the float
        weights directly, so no integer scaling is applied. `inputs` may be a single path or a list
        of paths; each may be a plain or gzipped (.gz) JSONL file, or a directory of such files.

        method="dataset" streams docs into a SeismicDataset and calls build_from_dataset (the API
        used by sentence-transformers). method="file" instead concatenates/decompresses the inputs
        into a temporary uncompressed JSONL file and calls the file-based build; use it as a
        fallback if build_from_dataset rejects a hyperparameter.
        """
        # resolve files/directories up front and fail loudly on unreadable input (this is experiment
        # code: we want a hard error, not a silently-skipped shard)
        inputs = _expand_inputs(inputs)

        seismic = _import_seismic()

        if variant == "standard":
            index_cls = seismic.SeismicIndex
        elif variant == "large_vocab":
            # needed for collections with more than 65k unique tokens (e.g. some multilingual LSR vocabs)
            index_cls = seismic.SeismicIndexLV
        else:
            raise ValueError(f"unknown Seismic index variant: {variant}")

        build_kwargs = {
            "n_postings": n_postings,
            "centroid_fraction": centroid_fraction,
            "summary_energy": summary_energy,
            "min_cluster_size": min_cluster_size,
            "max_fraction": max_fraction,
            "nknn": nknn,
            "batched_indexing": batched_indexing,
            "num_threads": THREADS,
        }

        if method == "dataset":
            string_type = seismic.get_seismic_string()
            dataset = seismic.SeismicDataset()
            for doc_id, vector in tqdm(_iter_docs(inputs), desc="seismic: adding documents", leave=False):
                dataset.add_document(
                    doc_id,
                    _to_seismic_strings(list(vector.keys()), string_type),
                    np.array(list(vector.values()), dtype=np.float32),
                )

            index = index_cls.build_from_dataset(dataset, **build_kwargs)
        elif method == "file":
            with _seismic_input(inputs) as build_input:
                index = index_cls.build(build_input, **build_kwargs)
        else:
            raise ValueError(f"unknown Seismic build method: {method}")

        print("Number of documents:", index.len)
        print("Avg number of non-zero components:", index.nnz / index.len)
        print("Dimensionality of the vectors:", index.dim)
        index.print_space_usage_byte()

        index.save(str(index_path))
        # seismic's save() appends INDEX_SUFFIX, so the on-disk filename differs from index_path
        print(f"saved index to: {index_path}{INDEX_SUFFIX}")
        return index
