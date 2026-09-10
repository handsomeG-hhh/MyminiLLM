from PIL import Image
from torch.utils.data import Dataset

class LRVInstructionDataset(Dataset):
    def __init__(
        self,
        data,
        processor,
        tokenizer,
        max_length=256
    ):
        self.data=data
        self.processor=processor
        self.tokenizer=tokenizer
        self.max_length=max_length
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        item=self.data[index]
        image=Image.open(item["image"]).convert("RGB")
        pixel_values=self.processor(
            images=image,
            return_tensors="pt"
        ).pixel_values[0]
        question=item["question"]
        answer=item["answer"]
        prompt=(
            "user"+
            question+
            "\nAssistant"+
            answer
        )
        tokens=self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids=tokens.input_ids[0]
        attention_mask=tokens.attention_mask[0]
        labels=input_ids.clone()
        labels[attention_mask==0]=-100