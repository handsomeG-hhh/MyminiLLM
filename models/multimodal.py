import torch
import torch.nn as nn


class MultiModalModel(nn.Module):
    def __init__(self, vision_encoder, projector, qwen, resampler=None):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.projector = projector
        self.qwen = qwen
        self.resampler = resampler

    def encode_image(self, pixel_values):
        with torch.no_grad():
            vision_outputs = self.vision_encoder(pixel_values)
        if self.resampler is not None:
            vision_outputs = self.resampler(vision_outputs)
        visual_tokens = self.projector(vision_outputs)
        return visual_tokens

    def forward(self, pixel_values, input_ids, attention_mask, labels):
        image_embeds = self.encode_image(pixel_values)
        text_embeds = self.qwen.get_input_embeddings()(input_ids)
        image_embeds = image_embeds.to(dtype=text_embeds.dtype)

        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)
        image_attention = torch.ones(
            image_embeds.shape[:2],
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        attention_mask = torch.cat([image_attention, attention_mask], dim=1)

        image_labels = torch.full(
            image_embeds.shape[:2],
            -100,
            dtype=labels.dtype,
            device=labels.device,
        )
        labels = torch.cat([image_labels, labels], dim=1)

        outputs = self.qwen(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs
