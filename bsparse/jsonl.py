import gzip
import json
from dataclasses import dataclass


@dataclass
class SparseRepresentations:
    ids: list[str]
    weights: list[dict[str, float]]
    term2id: dict[str, int] = None


def dict2jsonl(ds, docids, output_fn, term2id=None):
    if output_fn.suffix == ".gz":
        outf = gzip.open(output_fn, "wt", encoding="utf-8")
    else:
        outf = open(output_fn, "wt", encoding="utf-8")

    for docid, d in zip(docids, ds):
        print(json.dumps({"id": docid, "vector": d}), file=outf)

    outf.close()


def jsonl2dict(fn):
    ds = []
    docids = []

    if fn.suffix == ".gz":
        f = gzip.open(fn, "rt", encoding="utf-8")
    else:
        f = fn.open("rt", encoding="utf-8")

    for line in f:
        d = json.loads(line)
        ds.append(d["vector"])
        docids.append(d["id"])

    f.close()

    return SparseRepresentations(ids=docids, weights=ds, term2id=None)
