import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor
from modelscope import snapshot_download
import os
import json
import time
from pathlib import Path


class VisionEncoder(nn.Module):
    def __init__(self, model_name="AI-ModelScope/clip-vit-large-patch14"):
        super().__init__()
        if os.path.isdir(str(model_name)):
            model_dir = str(model_name)
        else:
            model_dir = snapshot_download(model_name)

        self.model = CLIPVisionModel.from_pretrained(model_dir)
        self.processor = CLIPImageProcessor.from_pretrained(model_dir)

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        return outputs.last_hidden_state
