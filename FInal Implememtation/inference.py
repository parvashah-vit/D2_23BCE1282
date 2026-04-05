from __future__ import annotations

import base64
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


ROOT_DIR = Path(__file__).resolve().parents[1]
BINARY_MODEL_PATH = ROOT_DIR / "binary_model_4.pth"
SEGMENTATION_MODEL_PATH = ROOT_DIR / "uNetModel_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


classification_transform = transforms.Compose(
    [
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

segmentation_transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


class CustomResNet50(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        self.backbone.fc = nn.Identity()
        self.rotation_head = nn.Sequential(
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.rotation_head(features)


class ResNetEncoder(nn.Module):
    def __init__(self, model: CustomResNet50) -> None:
        super().__init__()
        self.encoder = nn.Sequential(*list(model.backbone.children())[:-2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class AutoEncoder(nn.Module):
    def __init__(self, classifier_model: CustomResNet50) -> None:
        super().__init__()
        self.encoder = ResNetEncoder(classifier_model)
        self.decoder = Decoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        return self.decoder(encoded)


@lru_cache(maxsize=1)
def get_models() -> tuple[CustomResNet50, AutoEncoder]:
    classifier = CustomResNet50().to(DEVICE)
    classifier.load_state_dict(torch.load(BINARY_MODEL_PATH, map_location=DEVICE))
    classifier.eval()

    segmentation_backbone = CustomResNet50().to(DEVICE)
    segmentation_backbone.load_state_dict(
        torch.load(BINARY_MODEL_PATH, map_location=DEVICE)
    )
    segmentation_model = AutoEncoder(segmentation_backbone).to(DEVICE)
    segmentation_model.load_state_dict(
        torch.load(SEGMENTATION_MODEL_PATH, map_location=DEVICE)
    )
    segmentation_model.eval()
    return classifier, segmentation_model


def _image_to_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{payload}"


def _build_overlay(source: Image.Image, mask_image: Image.Image) -> Image.Image:
    base = source.convert("RGBA")
    grayscale_mask = mask_image.convert("L")
    overlay = Image.new("RGBA", base.size, (255, 82, 82, 0))
    overlay.putalpha(grayscale_mask.point(lambda px: 150 if px > 0 else 0))
    return Image.alpha_composite(base, overlay).convert("RGB")


def run_inference(image_bytes: bytes) -> dict[str, object]:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    original_size = image.size

    classifier, segmentation_model = get_models()

    with torch.inference_mode():
        classification_tensor = classification_transform(image).unsqueeze(0).to(DEVICE)
        classification_logits = classifier(classification_tensor)
        tumour_probability = torch.sigmoid(classification_logits).item()

        segmentation_tensor = segmentation_transform(image).unsqueeze(0).to(DEVICE)
        segmentation_logits = segmentation_model(segmentation_tensor)
        segmentation_probability = torch.sigmoid(segmentation_logits)[0, 0].cpu()

    tumour_detected = tumour_probability >= 0.5
    confidence = tumour_probability if tumour_detected else 1.0 - tumour_probability

    mask_tensor = (segmentation_probability >= 0.5).to(torch.uint8) * 255
    mask_image = Image.fromarray(mask_tensor.numpy(), mode="L").resize(
        original_size,
        Image.Resampling.NEAREST,
    )
    segmented_overlay = _build_overlay(image, mask_image)

    return {
        "status": "Detected" if tumour_detected else "Not Detected",
        "tumour_probability": round(tumour_probability * 100, 2),
        "confidence": round(confidence * 100, 2),
        "segmented_image": _image_to_data_url(segmented_overlay),
        "mask_image": _image_to_data_url(mask_image),
    }
