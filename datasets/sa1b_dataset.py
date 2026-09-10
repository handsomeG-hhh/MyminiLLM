from torch.utils.data import Dataset
from PIL import Image

class PretrainedDataset(Dataset):
    def __init__(self,data,processor,tokenizer):
        self.data=data
        self.processor=processor
        self.tokenizer=tokenizer
    def __getitem__(self,idx):
        item=self.data[idx]
        image=Image.open(item["image"]).convert("RGB")
        caption=item["caption"]
        pixel_values=self.processor(images=image,return_tensors="pt").pixel_values[0]
        text=caption
        tokens=self.tokenizer(text,return_tensors="pt",padding="max_length",truncation=True,max_length=128)
        return {
            "pixel_values":pixel_values,
            "input_ids":tokens.input_ids[0],
            "attention_mask":tokens.attention_mask[0],
            "labels":tokens.input_ids[0].clone()
        }
