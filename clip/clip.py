from transformers import CLIPProcessor, CLIPModel
import torch
from torch import nn
from PIL import Image

class CLIPModelWrapper(nn.Module):
    def __init__(self, model_name="./"):
        super().__init__()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.clip_model = CLIPModel.from_pretrained(model_name).to("cuda")
        self.clip_model.eval()  

    @property
    def dtype(self):
        return self.clip_model.visual.conv1.weight.dtype

    def encode_image(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to("cuda")
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)  # (B, 512)
            image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-10)
        return image_features

    def encode_text(self, text):
        inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True, max_length=77).to("cuda")
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)  # (B, 512)
            text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-10)
        return text_features
