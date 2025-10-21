import pytest
import torch

from bsparse.convert import (
    dict2vec,
    vec2dict,
)


def test_vec2dict_max_terms():
    vocab = {0: "zero", 1: "one", 2: "two"}
    assert vec2dict(torch.tensor([[1, 2, 3]]), vocab, max_terms=1) == [{"two": 3}]
    assert vec2dict(torch.tensor([[0, 0, 2]]), vocab, max_terms=2) == [{"two": 2}]
    assert vec2dict(torch.tensor([[1, 1, 0]]), vocab, max_terms=3) == [{"zero": 1, "one": 1}]

    # array is larger than the vocab size
    with pytest.raises(Exception):
        vec2dict(torch.tensor([[0, 1, 0, 0]]), vocab, max_terms=1)

    # array is smaller than the vocab size
    with pytest.raises(Exception):
        vec2dict(torch.tensor([[0, 0]]), vocab, max_terms=1)


def test_vec2dict_all():
    vocab = {0: "zero", 1: "one", 2: "two"}
    assert vec2dict(torch.tensor([[1, 2, 3]]), vocab, max_terms=None) == [{"zero": 1, "one": 2, "two": 3}]
    assert vec2dict(torch.tensor([[0, 0, 2]]), vocab, max_terms=None) == [{"two": 2}]
    assert vec2dict(torch.tensor([[1, 1, 0]]), vocab, max_terms=None) == [{"zero": 1, "one": 1}]

    # array is larger than the vocab size
    with pytest.raises(Exception):
        vec2dict(torch.tensor([[0, 1, 0, 0]]), vocab, max_terms=None)

    # array is smaller than the vocab size
    with pytest.raises(Exception):
        vec2dict(torch.tensor([[0, 0]]), vocab, max_terms=None)


def test_single_dict2vec():
    vocab = {0: "zero", 1: "one", 2: "two"}
    rev_vocab = {v: k for k, v in vocab.items()}
    assert all(dict2vec({"two": 2}, rev_vocab) == torch.tensor([0, 0, 2]))
    assert all(dict2vec({"zero": 1, "one": 1}, rev_vocab) == torch.tensor([1, 1, 0]))

    # dict contains terms that are not in the vocab
    with pytest.raises(Exception):
        dict2vec({"one": 1, "invalid": 1}, vocab)


def test_batch_dict2vec():
    vocab = {0: "zero", 1: "one", 2: "two"}
    rev_vocab = {v: k for k, v in vocab.items()}
    assert (dict2vec([{"two": 2}, {"zero": 1}], rev_vocab) == torch.tensor([[0, 0, 2], [1, 0, 0]])).all()
    assert (dict2vec([{"zero": 1, "one": 1}], rev_vocab) == torch.tensor([[1, 1, 0]])).all()

    # dict contains terms that are not in the vocab
    with pytest.raises(Exception):
        dict2vec([{"one": 1, "invalid": 1}], vocab)
