import itertools
from abc import ABC, abstractmethod
from functools import partial

import safetensors
import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from bsparse.convert import vec2dict
from bsparse.utils import get_torch_device


class Model(ABC):
    @classmethod
    @abstractmethod
    def add_arguments(cls, parser) -> None:
        """Add model-specific arguments to the parser"""
        pass


class SpladeModel(Model):
    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument(
            "--model-name", type=str, default="naver/splade_v2_distil", help="Model name or path (default: %(default)s)"
        )
        parser.add_argument(
            "--tokenizer-name", type=str, default=None, help="Tokenizer name or path (default: use model_name if not specified)"
        )
        parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length (default: %(default)s)")
        parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing (default: %(default)s)")
        parser.add_argument("--max-terms", type=int, default=None, help="Maximum number of terms (default: no limit)")

    def __init__(self, config):
        self.model_name = config.model_name
        self.tokenizer_name = config.model_name if not config.tokenizer_name else config.tokenizer_name
        self.max_length = config.max_length
        self.batch_size = config.batch_size
        self.max_terms = config.max_terms
        self.device = get_torch_device()
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True)
        self.tokenize = partial(
            self.tokenizer,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

        self.term2id = self.tokenizer.vocab
        self.id2term = {idx: term for term, idx in self.term2id.items()}

    def encode(self, data):
        ids = []
        encoded = []

        data_iter = iter(data)
        with torch.no_grad():
            for _ in tqdm(
                range(0, len(data), self.batch_size),
                desc="spladev2.encode",
                leave=True,
            ):
                id_batch, text_batch = zip(*list(itertools.islice(data_iter, self.batch_size)))
                assert text_batch

                batch_inputs = self.tokenize(text_batch)
                ids.extend(id_batch)

                token_batch = {k: v.to(self.device) for k, v in batch_inputs.items()}
                embeddings = self.encode_batch(token_batch)
                encoded.extend(embeddings.to("cpu"))

        dicts = vec2dict(torch.vstack(encoded), id2term=self.id2term, max_terms=self.max_terms)
        return ids, dicts

    def encode_batch(self, tokens):
        logits = self.model(**tokens)["logits"]
        term_scores = torch.log1p(torch.relu(logits)) * tokens["attention_mask"].unsqueeze(-1)
        pooled = torch.max(term_scores, dim=1).values
        return pooled


# SpladeModel with passage splitting
class SpladePsgModel(SpladeModel):
    @classmethod
    def add_arguments(cls, parser):
        super().add_arguments(cls, parser)
        parser.add_argument("--stride", type=int, default=256, help="Stride when creating passages (default: %(default)s)")

    def __init__(self, config):
        super().__init__(config)
        self.stride = config.stride
        self.tokenize = partial(
            self.tokenizer,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            stride=self.stride,
            return_overflowing_tokens=True,
            return_tensors="pt",
        )

    def encode(self, data):
        ids = []
        encoded = []

        # TODO change padding=longest ? maybe should be an option in case of model incompatibilities
        # TODO add option to try compiling model (I think this would go in the encode class?)
        # TODO factor this out into a separate tokenize_batch method, then unify SpladeModel and SpladePsgModel (e.g., set stride=0 for FirstP)
        data_iter = iter(data)
        with torch.no_grad():
            for _ in tqdm(
                range(0, len(data), self.batch_size),
                desc="encode",
                leave=True,
            ):
                id_batch, text_batch = zip(*list(itertools.islice(data_iter, self.batch_size)))
                assert text_batch

                batch_inputs = self.tokenize(text_batch)
                overflow_to_sample_mapping = batch_inputs.pop("overflow_to_sample_mapping", None)
                assert overflow_to_sample_mapping is not None, "tokenizer API changed???"

                new_ids = [f"{id_batch[orig_idx]}::psg{i}" for i, orig_idx in enumerate(overflow_to_sample_mapping.tolist())]
                ids.extend(new_ids)

                token_batch = {k: v.to(self.device) for k, v in batch_inputs.items()}
                embeddings = self.encode_batch(token_batch)
                encoded.extend(embeddings.to("cpu"))

        dicts = vec2dict(torch.vstack(encoded), id2term=self.id2term, max_terms=self.max_terms)
        return ids, dicts


class MultiLSRModel(Model):
    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path or model identifier")
        parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length (default: %(default)s)")
        parser.add_argument("--stride", type=int, default=256, help="Stride for sliding window processing (default: %(default)s)")
        parser.add_argument("--batch-size", type=int, default=128, help="Batch size for processing (default: %(default)s)")
        parser.add_argument("--max-terms", type=int, default=None, help="Maximum number of terms (default: no limit)")
        parser.add_argument("--base-model", type=str, default="FacebookAI/xlm-roberta-large")

    def __init__(self, config):
        self.max_length = config.max_length
        self.stride = config.stride
        self.batch_size = config.batch_size
        self.max_terms = config.max_terms
        self.device = get_torch_device()
        self.base_model = config.base_model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model, use_fast=True, padding_side="left" if "Qwen3" in self.base_model else "right"
        )
        self.tokenize = partial(
            self.tokenizer,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            stride=self.stride,
            return_overflowing_tokens=True,
            return_tensors="pt",
        )

        from multilsr.model import MultilingualModel as TextModel

        dense_size = 1024
        if "Qwen3-Embedding-4B" in self.base_model:
            dense_size = 2560

        self.model = TextModel(self.base_model, dense_size)
        config.ckpt = config.ckpt + "/model.safetensors" if not config.ckpt.endswith(".safetensors") else config.ckpt
        missing, unexpected = safetensors.torch.load_model(self.model, config.ckpt, strict=False)
        missing = {k for k in missing if not k.startswith("target_encoder")}
        assert not missing, missing
        assert not unexpected, unexpected

        self.model = self.model.to(self.device)
        self.model.eval()

        self.term2id = AutoTokenizer.from_pretrained("distilbert-base-uncased", use_fast=True).vocab
        self.id2term = {idx: term for term, idx in self.term2id.items()}

    def encode(self, data):
        ids = []
        encoded = []

        data_iter = iter(data)
        with torch.no_grad():
            for _ in tqdm(
                range(0, len(data), self.batch_size),
                desc="encode",
                leave=True,
            ):
                id_batch, text_batch = zip(*list(itertools.islice(data_iter, self.batch_size)))
                assert text_batch

                batch_inputs = self.tokenize(text_batch)
                overflow_to_sample_mapping = batch_inputs.pop("overflow_to_sample_mapping", None)
                assert overflow_to_sample_mapping is not None, "tokenizer API changed???"

                # TODO do something like this to make psg ids relative to the doc:
                # [(idx, psgid) for idx in ids for psgid in range(collections.Counter(overflow_to_sample_mapping.tolist())[idx])]

                new_ids = [f"{id_batch[orig_idx]}::psg{i}" for i, orig_idx in enumerate(overflow_to_sample_mapping.tolist())]
                ids.extend(new_ids)

                token_batch = {k: v.to(self.device) for k, v in batch_inputs.items()}
                embeddings = self.encode_batch(token_batch)
                encoded.extend(embeddings.to("cpu"))

        dicts = vec2dict(torch.vstack(encoded), id2term=self.id2term, max_terms=self.max_terms)
        return ids, dicts

    def encode_batch(self, tokens):
        return self.model.encode(**tokens)
