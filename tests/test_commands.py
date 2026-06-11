import argparse

import pytest


def _make_search(**kwargs):
    # importing bsparse.commands requires trecrun, which is optional in the test env
    pytest.importorskip("trecrun")
    from bsparse.commands import Search

    defaults = {"backend": "anserini", "scale": None, "query_cut": None, "heap_factor": None}
    defaults.update(kwargs)
    return Search(argparse.Namespace(**defaults))


def test_search_accepts_args_for_selected_backend():
    # args belonging to the chosen backend are fine; so is passing nothing
    _make_search(backend="anserini")._check_backend_args()
    _make_search(backend="anserini", scale=80)._check_backend_args()
    _make_search(backend="seismic", query_cut=12, heap_factor=0.9)._check_backend_args()


def test_search_rejects_args_for_other_backend():
    with pytest.raises(ValueError, match="seismic backend"):
        _make_search(backend="anserini", heap_factor=0.9)._check_backend_args()

    with pytest.raises(ValueError, match="seismic backend"):
        _make_search(backend="anserini", query_cut=5)._check_backend_args()

    with pytest.raises(ValueError, match="anserini backend"):
        _make_search(backend="seismic", scale=80)._check_backend_args()


def test_load_qrels_from_trec_file(tmp_path):
    pytest.importorskip("trecrun")
    from bsparse.commands import load_qrels

    qrels_fn = tmp_path / "qrels.txt"
    qrels_fn.write_text("q1 0 d1 1\nq1 0 d2 0\n\nq2 Q0 d1 2\n")
    assert load_qrels(str(qrels_fn)) == {"q1": {"d1": 1, "d2": 0}, "q2": {"d1": 2}}


def test_load_qrels_rejects_invalid_file(tmp_path):
    pytest.importorskip("trecrun")
    from bsparse.commands import load_qrels

    qrels_fn = tmp_path / "qrels.txt"
    qrels_fn.write_text("q1 0 d1 1\nq1 d2 0\n")
    with pytest.raises(ValueError, match="line 2"):
        load_qrels(str(qrels_fn))

    qrels_fn.write_text("q1 0 d1 relevant\n")
    with pytest.raises(ValueError, match="line 1"):
        load_qrels(str(qrels_fn))


def test_load_qrels_rejects_dataset_without_qrels(monkeypatch):
    pytest.importorskip("trecrun")
    import sys
    import types

    from bsparse.commands import load_qrels

    # stub ir_datasets with a dataset that exists but has no relevance judgments
    fake_irds = types.SimpleNamespace(load=lambda name: types.SimpleNamespace(has_qrels=lambda: False))
    monkeypatch.setitem(sys.modules, "ir_datasets", fake_irds)

    with pytest.raises(ValueError, match="has no qrels"):
        load_qrels("beir/nfcorpus")


def test_load_qrels_rejects_unknown_name(tmp_path):
    pytest.importorskip("trecrun")
    pytest.importorskip("ir_datasets")
    from bsparse.commands import load_qrels

    # not an existing file, and not a known ir_datasets name
    with pytest.raises(ValueError, match="neither an existing file nor a known ir_datasets name"):
        load_qrels(str(tmp_path / "no-such-qrels.txt"))
