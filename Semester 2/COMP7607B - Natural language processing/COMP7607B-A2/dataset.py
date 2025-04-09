import json
import os

import torch
from torch.utils.data import IterableDataset, get_worker_info

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEFAULT_VAL_SIZE = 1000


class _IterableDataset(IterableDataset):
    def __init__(self, data_iter, tokenizer, max_length, total_size, split_point, is_train):
        super().__init__()
        self.data_iter = data_iter
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.total_size = total_size
        self.split_point = split_point
        self.is_train = is_train
        self.condition = self.train_condition if self.is_train else self.val_condition
        self.size = self.total_size - self.split_point if self.is_train else self.split_point

    def train_condition(self, idx: int) -> bool:
        return idx > self.split_point

    def val_condition(self, idx: int) -> bool:
        return idx <= self.split_point

    def get_sources(self):
        raise NotImplementedError

    def get_references(self):
        raise NotImplementedError

    @property
    def samples(self):
        for idx, line in enumerate(self.data_iter):
            if self.condition(idx):
                yield json.loads(line.strip())

    def _inner(self, sample):
        raise NotImplementedError

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            for sample in self.samples:
                yield self._inner(sample)
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            for idx, sample in enumerate(self.samples):
                if idx % num_workers == worker_id:
                    yield self._inner(sample)

    def __len__(self):
        return self.size


class _PretrainIterableDataset(_IterableDataset):
    def _inner(self, sample):
        text = f"{self.tokenizer.bos_token}{str(sample['text'])}{self.tokenizer.eos_token}"
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding.input_ids.squeeze()
        loss_mask = input_ids != self.tokenizer.pad_token_id

        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
        return X, Y, loss_mask


class _SFTIterableDataset(_IterableDataset):
    def __init__(self, data_iter, tokenizer, max_length, total_size, split_point, is_train):
        super().__init__(data_iter, tokenizer, max_length, total_size, split_point, is_train)
        self.bos_id = tokenizer("<s>assistant\n", add_special_tokens=False).input_ids
        self.eos_id = tokenizer("</s>\n", add_special_tokens=False).input_ids
        self.prompt_length = 65

    def _create_chat_prompt(self, conversations):
        """Build dialogue in ChatML format"""
        messages = []
        for i, turn in enumerate(conversations):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": turn["content"]})
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i : i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end : end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def get_sources(self):
        sources = []
        for sample in self.samples:
            conversations = sample["conversations"]
            source = conversations[0]["content"][self.prompt_length :].strip()
            sources.append(source)
        return sources

    def get_references(self):
        references = []
        for sample in self.samples:
            conversations = sample["conversations"]
            reference = conversations[1]["content"].strip()
            references.append(reference)
        return references

    def get_messages_lst(self):
        return [[sample["conversations"][0]] for sample in self.samples]

    def _inner(self, sample):
        # Build dialogue prompt
        prompt = self._create_chat_prompt(sample["conversations"])
        input_ids = self.tokenizer(prompt).input_ids[: self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        # Generate dynamic loss mask
        loss_mask = self._generate_loss_mask(input_ids)

        # Build training data
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)

        return X, Y, loss_mask


class _DPOIterableDataset(_IterableDataset):
    def __init__(self, data_iter, tokenizer, max_length, total_size, split_point, is_train):
        super().__init__(data_iter, tokenizer, max_length, total_size, split_point, is_train)
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer("<s>assistant\n", add_special_tokens=False).input_ids
        self.eos_id = tokenizer("</s>\n", add_special_tokens=False).input_ids
        self.prompt_length = 65

    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i : i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end : end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def get_sources(self):
        sources = []
        for sample in self.samples:
            chosen = sample["chosen"]
            source = chosen[0]["content"][self.prompt_length :].strip()
            sources.append(source)
        return sources

    def get_references(self):
        references = []
        for sample in self.samples:
            chosen = sample["chosen"]
            reference = chosen[1]["content"].strip()
            references.append(reference)
        return references

    def get_messages_lst(self):
        return [[sample["chosen"][0]] for sample in self.samples]

    def _inner(self, sample):
        chosen = sample["chosen"]  # A list containing multiple {role, content} pairs
        rejected = sample["rejected"]  # Same as above
        chosen_prompt = self.tokenizer.apply_chat_template(chosen, tokenize=False, add_generation_prompt=False)
        rejected_prompt = self.tokenizer.apply_chat_template(rejected, tokenize=False, add_generation_prompt=False)
        chosen_encoding = self.tokenizer(
            chosen_prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )

        chosen_input_ids = chosen_encoding["input_ids"]
        chosen_loss_mask = self._generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding["input_ids"]
        rejected_loss_mask = self._generate_loss_mask(rejected_input_ids)
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            "x_chosen": x_chosen,
            "y_chosen": y_chosen,
            "mask_chosen": mask_chosen,
            "x_rejected": x_rejected,
            "y_rejected": y_rejected,
            "mask_rejected": mask_rejected,
        }


class DatasetBase:
    _iterable_dataset_class: type[_IterableDataset]

    def _get_total_size(self) -> int:
        with open(self.file_path, encoding="utf-8") as f:
            for i, _ in enumerate(f):
                pass
        return i + 1

    def _setup(self):
        # Count the number of lines in the file
        total_size = self._get_total_size()

        # Calculate validation size
        val_size = min(DEFAULT_VAL_SIZE, total_size // 10)

        # Create the train and validation datasets
        self.train_ds = self._iterable_dataset_class(
            self,
            self.tokenizer,
            self.max_length,
            total_size,
            val_size,
            is_train=True,
        )
        self.val_ds = self._iterable_dataset_class(
            self,
            self.tokenizer,
            self.max_length,
            total_size,
            val_size,
            is_train=False,
        )

    def __init__(self, file_path, tokenizer, max_length=512):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._setup()

    def __iter__(self):
        with open(self.file_path, encoding="utf-8") as f:
            yield from f


class PretrainDataset(DatasetBase):
    _iterable_dataset_class = _PretrainIterableDataset


class SFTDataset(DatasetBase):
    _iterable_dataset_class = _SFTIterableDataset


class DPODataset(DatasetBase):
    _iterable_dataset_class = _DPOIterableDataset
