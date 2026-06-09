import argparse

import pytest


def _make_search(**kwargs):
    # importing bsparse.commands requires trecrun/ir_datasets, which are optional in the test env
    pytest.importorskip("trecrun")
    pytest.importorskip("ir_datasets")
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
