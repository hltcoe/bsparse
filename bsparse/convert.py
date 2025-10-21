import torch

from bsparse.jsonl import SparseRepresentations, dict2jsonl, jsonl2dict


def load_dict(fn) -> SparseRepresentations:
    return jsonl2dict(fn)


def save_dict(ds, docids, output_fn, term2id=None):
    return dict2jsonl(ds, docids, output_fn, term2id=term2id)


def vec2dict(vecs, id2term, max_terms=None):
    # take sparse vectors as input and output a dict
    # use vocab to map sparse dims to terms

    if len(vecs.shape) == 1:
        vecs = vecs.unsqueeze(0)
        squeeze = True
    elif len(vecs.shape) == 2:
        squeeze = False
    else:
        raise ValueError("len(vecs.shape) must be 1 or 2")

    if vecs.shape[-1] != len(id2term):
        raise ValueError("vecs.shape[-1] is not equal to the id2term vocab size")

    if not max_terms:
        max_terms = vecs.shape[-1]

    weights, indices = vecs.topk(max_terms, dim=1)
    # indices = indices.to("cpu").tolist()
    # weights = weights.to("cpu").tolist()

    batch_reps = [
        {id2term[termid]: weight for termid, weight in zip(bi.tolist(), bw.tolist()) if weight > 0}
        for bi, bw in zip(indices, weights)
    ]
    if squeeze:
        return batch_reps[0]
    else:
        return batch_reps


def dict2vec(ds, term2id):
    # take one or more dicts as input and output a numpy sparse vector
    # use vocab to map terms to sparse dims
    if isinstance(ds, dict):
        return _single_dict2vec(ds, term2id)

    return torch.stack([_single_dict2vec(d, term2id) for d in ds])


def _single_dict2vec(d, term2id):
    terms, weights = zip(*d.items())
    termids = [term2id[term] for term in terms]
    vec = torch.zeros(len(term2id), dtype=torch.float32)
    vec[termids] = torch.tensor(weights).float()
    return vec


def token_ids_to_binary_vec(input_ids, attention_mask, special_tokens_mask, vocab_size):
    binary_ids = torch.ones_like(input_ids, dtype=torch.float) * attention_mask * (1 - special_tokens_mask)
    batch_size = binary_ids.shape[0]
    sparse_rep = torch.zeros((batch_size, vocab_size), device=binary_ids.device).scatter_reduce_(
        1, input_ids, binary_ids, reduce="amax"
    )
    return sparse_rep


def merge_reps(reps_list: list[SparseRepresentations], aggregate="max") -> SparseRepresentations:
    term2id = None
    docs = {}

    for reps in reps_list:
        if term2id is None:
            term2id = reps.term2id
        else:
            for term, tid in reps.term2id.items():
                if term not in term2id:
                    term2id[term] = len(term2id)

        for docid, ws in zip(reps.ids, reps.weights):
            if docid not in docs:
                docs[docid] = ws
            else:
                for term, w in ws.items():
                    docs[docid][term] = max(w, docs[docid].get(term, 0))

    docids, ds = zip(*docs.items())
    return SparseRepresentations(ids=docids, weights=ds, term2id=term2id)
