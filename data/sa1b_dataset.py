from torch.utils.data import Dataset
from PIL import Image
import requests
from io import BytesIO
import json
import hashlib 
from pathlib import Path
import ast
from torchvision.io import image

CACHE_DIR = Path(__file__).resolve().parent.parent/"data_cache"/"images"
CACHE_DIR.mkdir(parents=True,exist_ok=True)

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
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                return Image.open(BytesIO(resp.content)).convert("RGB")
            except Exception as err:
                last_err = err
        raise RuntimeError(f"image download failed: {url}") from last_err

    def __getitem__(self, idx):
        item = self.data[idx]
        url=item["url"]
        cap=item["cap_seg"]
        if isinstance(cap,str):
            try:
                cap=json.loads(cap)
            except json.JSONDecodeError:
                cap=ast.literal_eval(cap)
        caption=cap["global_caption"]
        try:
            img_path=CACHE_DIR / (hashlib.md5(url.encode()).hexdigest()+".jpg")
            try:
                if img_path.exists():
                    image=Image.open(img_path).convert("RGB")
                else:
                    image=self._download_image(url)
                    image.save(img_path)
            except Exception:
                return self[(idx+1)%len(self)]
        except Exception:
            # SA1B 的图源偶有死链：跳过换下一条，避免 10 万条里一条挂掉整个训练
            return self[(idx + 1) % len(self)]

        pixel_values = self.processor(
            images=image, return_tensors="pt"
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
