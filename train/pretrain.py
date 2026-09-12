import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.vision_encoder import VisionEncoder
from models.projector import projector
from models.multimodal import MultiModalModel
from data.sa1b_dataset import PretrainedDataset
from data.collator import MultiModalCollator
from modelscope import snapshot_download
from datasets import load_dataset as hf_load_dataset
import argparse
import os

# 调试用小模型；确认能跑通后再换更大的
VISION_MODEL = "AI-ModelScope/clip-vit-large-patch14"
QWEN_MODEL = "Qwen/Qwen2.5-0.5B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# None = 跑完整 epoch（10 万条样本）；调试通路时可临时设成 10
max_steps = None

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_steps", type=int, default=None, help="None=跑全量, 设10=通路测试")
    p.add_argument("--num_samples", type=int, default=100000)
    p.add_argument("--save_dir", type=str, default="./checkpoints")
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()
args=parse_args()
NUM_SAMPLES = args.num_samples 

# vision_dir=snapshot_download(VISION_MODEL)
qwen_dir=snapshot_download(QWEN_MODEL)

tokenizer = AutoTokenizer.from_pretrained(qwen_dir, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

vision_encoder = VisionEncoder(VISION_MODEL)
qwen = AutoModelForCausalLM.from_pretrained(
    qwen_dir,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    trust_remote_code=True,
)
proj = projector(
    vision_dim=vision_encoder.model.config.hidden_size,
    hidden_dim=qwen.config.hidden_size,
)
model = MultiModalModel(vision_encoder, proj, qwen)
model = model.to(DEVICE)

# 预训练：冻结 LLM 与视觉编码器，只训练 projector
for param in model.qwen.parameters():
    param.requires_grad = False
for param in model.vision_encoder.parameters():
    param.requires_grad = False

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4,
    weight_decay=0.01,
)

# 通路检测：先探 1 个 parquet 测行数，再按需补齐分片，避免整库 CDN 超时
# 证据: Read timed out host=cdn-lfs-cn-1.modelscope.cn during full MsDataset.load
from modelscope.hub.snapshot_download import dataset_snapshot_download
import math
import pyarrow.parquet as pq

  # 目标训练样本数：想改规模只动这一个数
shard_dir = ROOT / "data_cache" / "sa1b_shards"

dataset_snapshot_download(
    dataset_id="Tongyi-DataEngine/SA1B-Dense-Caption",
    local_dir=str(shard_dir),
    allow_file_pattern=[f"train-{i:04d}-of-0724-*.parquet" for i in range(20)],
)
_probe = sorted(shard_dir.rglob("*.parquet"))
if not _probe:
    raise FileNotFoundError(f"No parquet under {shard_dir}; probe download failed")
rows_per_shard = max(pq.ParquetFile(p).metadata.num_rows for p in _probe)
NUM_SHARDS = math.ceil(NUM_SAMPLES / rows_per_shard)
print(f"rows/shard={rows_per_shard} -> download {NUM_SHARDS} shards for ~{NUM_SAMPLES} samples")

dataset_snapshot_download(
    dataset_id="Tongyi-DataEngine/SA1B-Dense-Caption",
    local_dir=str(shard_dir),
    allow_file_pattern=[f"train-{i:04d}*" for i in range(NUM_SHARDS)],
)

parquet_files = sorted(str(p) for p in shard_dir.rglob("*.parquet"))
if not parquet_files:
    raise FileNotFoundError(f"No parquet under {shard_dir}; shard download failed")
data=hf_load_dataset("parquet",data_files=parquet_files,split="train")

if not  parquet_files:
    raise FileNotFoundError(f"No parquet under {shard_dir}; shard download failed")

data = hf_load_dataset("parquet", data_files=parquet_files, split="train")
print(f"Total rows available in downloaded shards: {len(data)}")

# ★ 选约 10 万条：固定 seed 洗牌后取前 NUM_SAMPLES 条（惰性索引，不复制进内存）
if len(data) > NUM_SAMPLES:
    data = data.shuffle(seed=42).select(range(NUM_SAMPLES))
print(f"Training samples after selection: {len(data)}")

dataset = PretrainedDataset(
    data=data,
    processor=model.vision_encoder.processor,
    tokenizer=tokenizer,
)

os.makedirs(args.save_dir, exist_ok=True)
dataloader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn=MultiModalCollator(),
    num_workers=args.num_workers,
    pin_memory=True,
    persistent_workers=True if args.num_workers > 0 else False,
)
model.train()
for epoch in range(1):
    for step, batch in enumerate(dataloader):
        if max_steps is not None and step >= max_steps:
            break
        pixel_values = batch["pixel_values"].to(DEVICE)
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"Epoch:{epoch} Step:{step}/{len(dataloader)} Loss:{loss.item():.4f}", flush=True)

        if step % args.save_every == 0 and step > 0:
            ckpt = os.path.join(args.save_dir, f"projector-epoch{epoch}-step{step}.pt")
            torch.save(model.projector.state_dict(), ckpt)
            print(f"saved {ckpt}", flush=True)
        if step % 50 == 0:
            print(f"Epoch:{epoch} Step:{step}/{len(dataloader)} Loss:{loss.item():.4f}", flush=True)

torch.save(model.projector.state_dict(), os.path.join(args.save_dir, "projector-final.pt"))
print("pretrain finished OK")