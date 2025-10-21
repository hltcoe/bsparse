import os
from functools import cache

import torch
from tqdm import tqdm


def psgid_to_docid(psgid):
    """Convert passage IDs (`doc123::psg2`) to their docID (`doc123`)"""
    return psgid.rsplit("::psg", 1)[0]


@cache
def get_torch_device():
    if torch.cuda.is_available():
        if os.environ.get("ASSERT_GPU_SPECIFIED", "false").lower() == "true" and not os.environ.get("CUDA_VISIBLE_DEVICES", ""):
            raise OSError("ASSERT_GPU_SPECIFIED=true but CUDA_VISIBLE_DEVICES is empty")

        return torch.device("cuda")
    else:
        if os.environ.get("ASSERT_GPU", "false").lower() == "true":
            raise OSError("ASSERT_GPU=true but cuda is not available")

        return torch.device("cpu")


def batch_encode(tokenized, model_encodef, device, batch_size: int = 128):
    encoded = []
    with torch.no_grad():
        for i in tqdm(
            range(0, len(tokenized["input_ids"]), batch_size),
            desc="bsparse.batch_encode",
            leave=False,
        ):
            tokenized_batch = {k: v[i : i + batch_size].to(device) for k, v in tokenized.items()}
            embed = model_encodef(**tokenized_batch)
            encoded.extend(embed.to("cpu"))
    return torch.vstack(encoded)


def batch_encode_untok(data, model_encodef, device, batch_size: int = 128):
    encoded = []
    with torch.no_grad():
        for i in tqdm(
            range(0, len(data), batch_size),
            desc="bsparse.batch_encode",
            leave=False,
        ):
            batch = data[i : i + batch_size].to(device)
            embed = model_encodef(batch)
            encoded.extend(embed.to("cpu"))
    return torch.vstack(encoded)


def token_ids_to_binary_vec(input_ids, attention_mask, special_tokens_mask, vocab_size):
    binary_ids = torch.ones_like(input_ids, dtype=torch.float) * attention_mask * (1 - special_tokens_mask)
    batch_size = binary_ids.shape[0]
    sparse_rep = torch.zeros((batch_size, vocab_size), device=binary_ids.device).scatter_reduce_(
        1, input_ids, binary_ids, reduce="amax"
    )
    return sparse_rep
