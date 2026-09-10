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
        # #region agent log
        _log_path = Path(__file__).resolve().parents[1] / "debug-429253.log"
        def _dbg(hypothesis_id, message, data, run_id="post-fix"):
            with open(_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "sessionId": "429253",
                    "runId": run_id,
                    "hypothesisId": hypothesis_id,
                    "location": "vision_encoder.py:__init__",
                    "message": message,
                    "data": data,
                    "timestamp": int(time.time() * 1000),
                }, ensure_ascii=False) + "\n")
        _dbg("A", "VisionEncoder entry", {
            "model_name": str(model_name),
            "is_abs_path": os.path.isabs(str(model_name)),
            "isdir": os.path.isdir(str(model_name)),
        })
        # #endregion

        if os.path.isdir(str(model_name)):
            model_dir = str(model_name)
        else:
            model_dir = snapshot_download(model_name)

        # #region agent log
        _dbg("A", "resolved local model_dir before from_pretrained", {
            "model_dir": str(model_dir),
            "isdir": os.path.isdir(str(model_dir)),
            "is_abs_path": os.path.isabs(str(model_dir)),
        })
        # #endregion

        self.model = CLIPVisionModel.from_pretrained(model_dir)
        self.processor = CLIPImageProcessor.from_pretrained(model_dir)

        # #region agent log
        _dbg("A", "from_pretrained finished from local dir", {"ok": True, "model_dir": str(model_dir)})
        # #endregion

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        return outputs.last_hidden_state
