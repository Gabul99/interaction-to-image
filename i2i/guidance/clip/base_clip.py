import torch
import torch.nn as nn
from .clip import clip
# from clip import clip
import torchvision
from PIL import Image

model_name = "ViT-B/16"
# model_name = "ViT-B/32"


def load_clip_to_cpu():
    url = clip._MODELS[model_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class CLIPEncoder(nn.Module):
    def __init__(self, need_ref=False, ref_path=None):
        super().__init__()
        self.clip_model = load_clip_to_cpu()
        self.clip_model.requires_grad = True
        self.preprocess = torchvision.transforms.Normalize(
            (0.48145466*2-1, 0.4578275*2-1, 0.40821073*2-1),
            (0.26862954*2, 0.26130258*2, 0.27577711*2)
        )
        if need_ref:
            self.to_tensor = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
            
            img = Image.open(ref_path).convert('RGB')
            image = img.resize((224, 224), Image.Resampling.BILINEAR)
            img = self.to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            self.ref = img

    def get_residual(self, image, text):
        text = clip.tokenize(text).cuda()
        image = torch.nn.functional.interpolate(image, size=224, mode='bicubic')
        image = self.preprocess(image)
        image_feature, _ = self.clip_model.encode_image_with_features(image)
        text_feature = self.clip_model.encode_text(text)
        text_feature = text_feature.repeat(image.shape[0], 1)
        return text_feature - image_feature

    def get_gram_matrix_residual(self, im1):
        im1 = torch.nn.functional.interpolate(im1, size=(224, 224), mode='bicubic')
        im1 = self.preprocess(im1)

        f1, feats1 = self.clip_model.encode_image_with_features(im1)
        f2, feats2 = self.clip_model.encode_image_with_features(self.ref)
        
        feat1 = feats1[2][1:, 0, :]
        feat2 = feats2[2][1:, 0, :]
        gram1 = torch.mm(feat1.t(), feat1)
        gram2 = torch.mm(feat2.t(), feat2)
        return gram1 - gram2



if __name__ == "__main__":
    m = CLIPEncoder().cuda()
    im1 = torch.randn((1, 3, 224, 224)).cuda()
    im2 = torch.randn((1, 3, 224, 224)).cuda()
    m.get_gram_matrix_residual(im1, im2)


# new class, e.g., in i2i/freedom/clip/base_clip.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoProcessor, SiglipVisionModel, SiglipTextModel

# class SigLIPEncoder(nn.Module):
#     def __init__(self, name="google/siglip2-so400m-patch14-384", device=None):
#         super().__init__()
#         self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model = AutoModel.from_pretrained(name).to(self.device).eval()
#         #self.vision = SiglipVisionModel.from_pretrained(name).to(self.device).eval()
#         #self.text = SiglipTextModel.from_pretrained(name).to(self.device).eval()
#         self.tokenizer = AutoTokenizer.from_pretrained(name)
#         proc = AutoProcessor.from_pretrained(name)
#         mean = torch.tensor(proc.image_processor.image_mean).view(1, -1, 1, 1).to(self.device)
#         std = torch.tensor(proc.image_processor.image_std).view(1, -1, 1, 1).to(self.device)
#         self.register_buffer("mean", mean)
#         self.register_buffer("std", std)
#         self.image_size = getattr(self.vision.config, "image_size", 384)

#     def get_residual(self, image, text):
#         # image: [-1, 1], BCHW -> SigLIP expected normalization
#         x = (image + 1) / 2
#         x = F.interpolate(x, size=self.image_size, mode="bicubic", align_corners=False)
#         x = (x - self.mean) / self.std
#         toks = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
#         with torch.no_grad():
#             txt = self.text(**toks).text_embeds
#             #last_hidden_state = self.text(**toks).last_hidden_state
#             #pooled_output = self.text(**toks).pooler_output  # pooled (EOS token) states
#         img = self.vision(pixel_values=x).image_embeds
#         # normalized features; residual drives cosine alignment
#         txt = txt / txt.norm(dim=-1, keepdim=True)
#         img = img / img.norm(dim=-1, keepdim=True)
#         return txt - img


# e.g., in i2i/freedom/clip/base_clip.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor


class SigLIPEncoder(nn.Module):
    """
    SigLIP2 encoder that returns a text–image embedding residual for guidance.

    - `image` is expected in [-1, 1], shape (B, C, H, W)
    - `text` is a str or list[str]
    - residual = normalized(text_features) - normalized(image_features)
      Gradients flow through the image branch (for diffusion guidance).
    """

    def __init__(self, name: str = "google/siglip-so400m-patch14-384", device=None):
        super().__init__()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # This will give you a Siglip2Model with get_text_features / get_image_features
        self.model = AutoModel.from_pretrained(name).to(self.device).eval()
        # Freeze weights so we only backprop through the inputs, not the encoder
        for p in self.model.parameters():
            p.requires_grad = False

        # Handles both image processor + tokenizer (Gemma tokenizer)
        self.processor = AutoProcessor.from_pretrained(name)
        # Cache image normalization stats and target size for differentiable preprocessing
        mean = torch.tensor(self.processor.image_processor.image_mean, dtype=torch.float32).view(1, -1, 1, 1)
        std = torch.tensor(self.processor.image_processor.image_std, dtype=torch.float32).view(1, -1, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.image_size = getattr(getattr(self.model, "config", None), "vision_config", None)
        self.image_size = getattr(self.image_size, "image_size", 384)

    def get_residual(self, image: torch.Tensor, text):
        """
        Compute text–image residual in the SigLIP2 embedding space.

        Returns:
            residual: (batch_size, dim) = text_feat_norm - image_feat_norm
        """
        # image: expected in [-1, 1]; convert to [0, 1] floats, resize and normalize (all differentiable)
        x = image.to(self.device)
        # Heuristic normalization to [0, 1]
        with torch.no_grad():
            x_min = float(x.min().detach())
            x_max = float(x.max().detach())
        if x_max > 1.5:
            # Likely [0, 255]
            x = x / 255.0
        elif x_min < -0.01 or x_max > 1.01:
            # Likely [-1, 1]
            x = (x.clamp(-1.0, 1.0) + 1.0) / 2.0
        else:
            # Already in [0, 1]
            x = x.clamp(0.0, 1.0)

        # Resize to model's expected square size and normalize with SigLIP stats
        x = torch.nn.functional.interpolate(
            x, size=(self.image_size, self.image_size), mode="bicubic", align_corners=False
        )
        x = (x - self.mean) / self.std

        # Text: str -> list[str]; SigLIP2 was trained on lowercased text.
        if isinstance(text, str):
            text = [text]
        text = [t.lower() for t in text]

        batch_size = x.shape[0]
        if len(text) == 1 and batch_size > 1:
            # Broadcast a single prompt across the image batch
            text = text * batch_size

        # Tokenize text only; keep our differentiable image path
        text_inputs = self.processor(
            text=text, padding="max_length", max_length=64, truncation=True, return_tensors="pt"
        )
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

        # Text features: keep them constant (no gradient needed through text)
        with torch.no_grad():
            text_features = self.model.get_text_features(**text_inputs
            )
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Image features: we want gradient w.r.t. image -> don't use no_grad here
        image_features = self.model.get_image_features(pixel_values=x)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Residual direction in embedding space (guidance signal)
        residual = text_features - image_features
        return residual