import torch

class MultiModalCollator:
    def __call__(
        self,
        batch
    ):
        pixel_values=torch.stack([
            item["pixel_values"]for item in batch
        ])
        input_ids=torch.stack([
            item["input_ids"]for item in batch
        ])
        attention_mask=torch.stack([
            item["attention_mask"]for item in batch
        ])
        labels=torch.stack([
            item["labels"]for item in batch
        ])
        return {
            "pixel_values":pixel_values,
            "input_ids":input_ids,
            "attention_mask":attention_mask,
            "labels":labels
        }
