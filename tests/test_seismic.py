import os
import types

import numpy as np
import pytest


def fake_seismic_module(**attrs):
    """A stand-in for the native 'seismic' module, so tests run without it installed."""
    return types.SimpleNamespace(get_seismic_string=lambda: "U30", **attrs)


def test_query_from_vectors_builds_arrays_and_maps_results(monkeypatch):
    """Seismic.query_from_vectors should convert each query dict into numpy component/value arrays
    and map the (query_id, score, doc_id) tuples back into per-query {doc_id: score} dicts, aligned
    with the input order. This exercises the conversion/mapping without the native 'seismic' lib."""
    import bsparse.seismic as seismic_mod
    from bsparse.seismic import Seismic

    # avoid importing the native package: stub the string type and capture the batch_search args
    monkeypatch.setattr(seismic_mod, "_import_seismic", fake_seismic_module)

    captured = {}

    class FakeIndex:
        def batch_search(self, queries_ids, query_components, query_values, **kwargs):
            captured["queries_ids"] = queries_ids
            captured["query_components"] = query_components
            captured["query_values"] = query_values
            captured["kwargs"] = kwargs
            # one result list per query; tuples are (query_id, score, doc_id)
            return [
                [("0", 2.5, "docA"), ("0", 1.0, "docB")],
                [("1", 3.0, "docC")],
            ]

    retriever = Seismic("ignored")
    retriever._index = FakeIndex()

    queries = [{"vector": {"dog": 1.5, "cat": 0.5}}, {"vector": {"fish": 2.0}}]
    results = retriever.query_from_vectors(queries, k=5, query_cut=7, heap_factor=0.9)

    assert results == [{"docA": 2.5, "docB": 1.0}, {"docC": 3.0}]

    # the native batch_search requires an ndarray[str] (not a list) for the query ids
    assert captured["queries_ids"].dtype == np.dtype("U30")
    assert list(captured["queries_ids"]) == ["0", "1"]
    assert captured["kwargs"]["k"] == 5
    assert captured["kwargs"]["query_cut"] == 7
    assert captured["kwargs"]["heap_factor"] == 0.9

    assert captured["query_components"][0].dtype == np.dtype("U30")
    assert list(captured["query_components"][0]) == ["dog", "cat"]
    assert captured["query_values"][0].dtype == np.float32
    assert list(captured["query_values"][1]) == pytest.approx([2.0])


def test_build_rejects_missing_input():
    from bsparse.seismic import Seismic

    # input validation happens before the native lib is imported, so this fails loudly even
    # without 'seismic' installed
    with pytest.raises(FileNotFoundError, match="readable file"):
        Seismic.build("does-not-exist.jsonl", "out_index")


def test_build_dataset_method_streams_docs(tmp_path, monkeypatch):
    """The default build method should stream docs (plain and .gz) into a SeismicDataset via
    add_document and pass the hyperparameters to build_from_dataset."""
    import gzip
    import json

    import bsparse.seismic as seismic_mod
    from bsparse.seismic import Seismic

    plain = tmp_path / "docs-0.jsonl"
    plain.write_text(
        json.dumps({"id": "a", "vector": {"dog": 1.5, "pet": 0.5}}) + "\n" + json.dumps({"id": "b", "vector": {"cat": 2.0}}),
        encoding="utf-8",
    )
    gzipped = tmp_path / "docs-1.jsonl.gz"
    with gzip.open(gzipped, "wt", encoding="utf-8") as f:
        f.write(json.dumps({"id": 3, "vector": {"fish": 3.0}}) + "\n")

    added = []
    captured = {}

    class FakeDataset:
        def add_document(self, doc_id, components, values):
            added.append((doc_id, components, values))

    class FakeBuiltIndex:
        len = 3
        nnz = 4
        dim = 4

        def print_space_usage_byte(self):
            pass

        def save(self, path):
            captured["saved_to"] = path

    class FakeSeismicIndex:
        @staticmethod
        def build_from_dataset(dataset, **kwargs):
            captured["dataset"] = dataset
            captured["kwargs"] = kwargs
            return FakeBuiltIndex()

    monkeypatch.setattr(
        seismic_mod,
        "_import_seismic",
        lambda: fake_seismic_module(SeismicDataset=FakeDataset, SeismicIndex=FakeSeismicIndex),
    )

    Seismic.build([plain, gzipped], tmp_path / "out_index", n_postings=123, nknn=7)

    # docs streamed in order, ids coerced to str, arrays use the seismic dtypes
    assert [d[0] for d in added] == ["a", "b", "3"]
    assert list(added[0][1]) == ["dog", "pet"]
    assert added[0][1].dtype == np.dtype("U30")
    assert added[0][2].dtype == np.float32
    assert list(added[2][2]) == pytest.approx([3.0])

    assert isinstance(captured["dataset"], FakeDataset)
    assert captured["kwargs"]["n_postings"] == 123
    assert captured["kwargs"]["nknn"] == 7
    assert captured["kwargs"]["centroid_fraction"] == seismic_mod.DEFAULT_CENTROID_FRACTION
    assert captured["saved_to"] == str(tmp_path / "out_index")


def test_build_dataset_method_fails_loudly_on_bad_json(tmp_path, monkeypatch):
    import bsparse.seismic as seismic_mod
    from bsparse.seismic import Seismic

    fn = tmp_path / "docs.jsonl"
    fn.write_text('{"id": "a", "vector": {"dog": 1.0}}\nnot json\n', encoding="utf-8")

    class FakeDataset:
        def add_document(self, *args):
            pass

    monkeypatch.setattr(
        seismic_mod,
        "_import_seismic",
        lambda: fake_seismic_module(SeismicDataset=FakeDataset, SeismicIndex=None),
    )

    with pytest.raises(ValueError, match="line 2"):
        Seismic.build(fn, tmp_path / "out_index")


def test_expand_inputs_handles_files_dirs_and_order(tmp_path):
    from bsparse.seismic import _expand_inputs

    explicit = tmp_path / "explicit.jsonl"
    explicit.write_text("{}\n", encoding="utf-8")

    docs_dir = tmp_path / "shards"
    docs_dir.mkdir()
    (docs_dir / "b.jsonl").write_text("{}\n", encoding="utf-8")
    (docs_dir / "a.jsonl.gz").write_text("{}\n", encoding="utf-8")
    (docs_dir / "ignore.txt").write_text("nope\n", encoding="utf-8")

    # a directory expands to its sorted .jsonl/.jsonl.gz files; non-matching files are ignored;
    # explicit args keep their position
    result = _expand_inputs([str(explicit), str(docs_dir)])
    assert result == [
        str(explicit),
        str(docs_dir / "a.jsonl.gz"),
        str(docs_dir / "b.jsonl"),
    ]


def test_expand_inputs_fails_loudly(tmp_path):
    from bsparse.seismic import _expand_inputs

    with pytest.raises(FileNotFoundError, match="file or directory"):
        _expand_inputs(str(tmp_path / "does-not-exist"))

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="no .jsonl"):
        _expand_inputs(str(empty_dir))


def test_resolve_index_path_accepts_prefix_or_filename(tmp_path):
    """seismic's save(path) writes to path + '.index.seismic' but load() wants the real filename, so
    searching with the same --index used at build time must resolve to the suffixed file."""
    from bsparse.seismic import _resolve_index_path

    prefix = tmp_path / "myindex"
    on_disk = tmp_path / "myindex.index.seismic"
    on_disk.write_bytes(b"")

    assert _resolve_index_path(str(prefix)) == str(on_disk)
    assert _resolve_index_path(str(on_disk)) == str(on_disk)

    with pytest.raises(FileNotFoundError, match="index.seismic"):
        _resolve_index_path(str(tmp_path / "missing"))


def test_seismic_input_single_plain_file_used_directly(tmp_path):
    from bsparse.seismic import _seismic_input

    fn = tmp_path / "docs.jsonl"
    fn.write_text('{"id": "a", "vector": {"dog": 1.0}}\n', encoding="utf-8")

    with _seismic_input([str(fn)]) as build_input:
        # a single uncompressed file is passed through untouched (no temp copy)
        assert build_input == str(fn)


def test_seismic_input_concatenates_and_decompresses(tmp_path):
    import gzip
    import json

    from bsparse.seismic import _seismic_input

    plain = tmp_path / "docs-0.jsonl"
    # no trailing newline on the last line, to check files are joined safely
    plain.write_text('{"id": "a", "vector": {"dog": 1.0}}\n{"id": "b", "vector": {"cat": 2.0}}', encoding="utf-8")

    gzipped = tmp_path / "docs-1.jsonl.gz"
    with gzip.open(gzipped, "wt", encoding="utf-8") as f:
        f.write('{"id": "c", "vector": {"fish": 3.0}}\n')

    with _seismic_input([str(plain), str(gzipped)]) as build_input:
        assert build_input != str(plain)  # a temp combined file
        lines = [json.loads(line) for line in open(build_input, encoding="utf-8")]

    ids = [d["id"] for d in lines]
    assert ids == ["a", "b", "c"]

    # the temp file is cleaned up on exit
    assert not os.path.exists(build_input)


@pytest.mark.parametrize("method", ["dataset", "file"])
def test_build_and_search_end_to_end(tmp_path, method):
    """End-to-end build + search against the real Seismic library (skipped if not installed)."""
    pytest.importorskip("seismic")

    import json

    from bsparse.seismic import Seismic

    docs = [
        {"id": "doc1", "vector": {"dog": 2.0, "pet": 1.0}},
        {"id": "doc2", "vector": {"cat": 2.0, "pet": 1.0}},
        {"id": "doc3", "vector": {"car": 2.0, "road": 1.0}},
    ]
    input_fn = tmp_path / "docs.jsonl"
    with open(input_fn, "wt", encoding="utf-8") as f:
        for d in docs:
            print(json.dumps(d), file=f)

    index_path = tmp_path / "index.seismic"
    Seismic.build(input_fn, index_path, n_postings=100, centroid_fraction=0.2, min_cluster_size=1, method=method)

    retriever = Seismic(str(index_path))
    results = retriever.query_from_vectors([{"vector": {"dog": 1.0, "pet": 1.0}}], k=3)

    assert len(results) == 1
    assert "doc1" in results[0]
    # the dog/pet query should rank doc1 (dog+pet) above doc3 (unrelated)
    assert results[0].get("doc1", 0) >= results[0].get("doc3", 0)
