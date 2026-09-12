from torch.utils.data import Dataset
from PIL import Image
import requests
from io import BytesIO
import json
import hashlib
from pathlib import Path
import ast

CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache" / "images"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_HEADERS = {"User-Agent": "Mozilla/5.0"}

class PretrainedDataset(Dataset):
    def __init__(self, data, processor, tokenizer, max_length=512):
        self.data = data
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def _download_image(self, url, retries=3):
        last_err = None
        for _ in range(retries):
            try:
                resp = requests.get(url, timeout=10, headers=_HEADERS)
                resp.raise_for_status()
                return Image.open(BytesIO(resp.content)).convert("RGB")
            except Exception as err:
                last_err = err
        raise RuntimeError(f"image download failed: {url}") from last_err

    def _load_image(self, url):
        img_path = CACHE_DIR / (hashlib.md5(url.encode()).hexdigest() + ".jpg")
        if img_path.exists():
            try:
                return Image.open(img_path).convert("RGB")
            except Exception:
                try:
                    img_path.unlink()
                except Exception:
                    pass
        pil_image = self._download_image(url)
        try:
            pil_image.save(img_path)
        except Exception:
            pass
        return pil_image

    def __getitem__(self, idx):
        for offset in range(len(self)):
            cur = (idx + offset) % len(self)
            try:
                item = self.data[cur]
                url = item["url"]
                cap = item["cap_seg"]
                if isinstance(cap, str):
                    try:
                        cap = json.loads(cap)
                    except json.JSONDecodeError:
                        cap = ast.literal_eval(cap)
                caption = cap["global_caption"]

                pil_image = self._load_image(url)
                pixel_values = self.processor(
                    images=pil_image, return_tensors="pt"
                ).pixel_values[0]
                tokens = self.tokenizer(
                    caption,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                )
                input_ids = tokens.input_ids[0]
                attention_mask = tokens.attention_mask[0]
                labels = input_ids.clone()
                labels[attention_mask == 0] = -100
                return {
                    "pixel_values": pixel_values,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            except Exception:
                continue
        raise RuntimeError("no valid sample found, check dataset urls")