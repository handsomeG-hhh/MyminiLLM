from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def load_qwen(model_name, torch_dtype=torch.float16, device_map=None):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"trust_remote_code": True, "torch_dtype": torch_dtype}
    if device_map is not None:
        kwargs["device_map"] = device_map

    qwen = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    return tokenizer, qwen
