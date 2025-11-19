"""
Model architecture and utilities for the Chlorella classification pipeline.

Handles:
- Pre-trained backbone loading (ResNet18, VGG11-BN)
- First conv layer adaptation for 4-channel input
- Classifier head replacement for 5 classes
- Model building utilities
"""

from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models


def build_backbone(
    architecture: str = "resnet18", pretrained: bool = True
) -> Tuple[nn.Module, int]:
    """
    Load pre-trained backbone model.

    Args:
        architecture: Model architecture name ("resnet18" or "vgg11_bn")
        pretrained: Whether to load ImageNet pre-trained weights

    Returns:
        Tuple of (model, feature_dim):
        - model: Backbone model (without final classifier)
        - feature_dim: Dimension of features before classifier

    Raises:
        ValueError: If architecture name is not supported
    """
    if architecture == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        feature_dim = model.fc.in_features
        # Remove final FC layer (will replace with custom head)
        model = nn.Sequential(*list(model.children())[:-1])
        return model, feature_dim

    elif architecture == "vgg11_bn":
        model = models.vgg11_bn(pretrained=pretrained)
        # VGG has two parts: features and classifier
        feature_extractor = model.features
        feature_dim = 512  # VGG11 outputs 512 features
        return feature_extractor, feature_dim

    else:
        raise ValueError(
            f"Unsupported architecture: {architecture}. " f"Supported: 'resnet18', 'vgg11_bn'"
        )


def adapt_first_conv_for_4ch(model: nn.Module, architecture: str = "resnet18") -> nn.Module:
    """
    Adapt first convolutional layer to accept 4-channel input.

    Modifies conv1 from (out_ch, 3, kernel, kernel) to (out_ch, 4, kernel, kernel)
    by copying pre-trained weights for first 3 channels and initializing
    the 4th channel with small random values.

    Args:
        model: Backbone model with standard 3-channel first conv
        architecture: Architecture name to determine which layer to modify

    Returns:
        Modified model with 4-channel first conv

    Raises:
        ValueError: If architecture is not supported
    """
    if architecture == "resnet18":
        # For ResNet wrapped in Sequential, first conv is at model[0] (the conv1 layer)
        old_conv = model[0]  # First child is conv1

        # Create new conv with 4 input channels
        new_conv = nn.Conv2d(
            in_channels=4,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        # Copy pre-trained weights for first 3 channels
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_conv.weight
            # Initialize 4th channel with small random values
            new_conv.weight[:, 3:, :, :] = torch.randn_like(new_conv.weight[:, 3:, :, :]) * 0.01

            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        # Replace first conv
        model[0] = new_conv

    elif architecture == "vgg11_bn":
        # For VGG, first conv is at model[0] (first layer in features)
        old_conv = model[0]

        # Create new conv with 4 input channels
        new_conv = nn.Conv2d(
            in_channels=4,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        # Copy pre-trained weights for first 3 channels
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_conv.weight
            # Initialize 4th channel with small random values
            new_conv.weight[:, 3:, :, :] = torch.randn_like(new_conv.weight[:, 3:, :, :]) * 0.01

            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        # Replace first conv
        model[0] = new_conv

    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    return model


def replace_classifier_head(
    model: nn.Module, architecture: str, feature_dim: int, num_classes: int = 5
) -> nn.Module:
    """
    Replace classifier head for custom number of output classes.

    Args:
        model: Backbone model
        architecture: Architecture name
        feature_dim: Dimension of features before classifier
        num_classes: Number of output classes

    Returns:
        Model with new classifier head
    """
    if architecture == "resnet18":
        # ResNet: Add global average pool + FC
        classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(feature_dim, num_classes)
        )
        model = nn.Sequential(model, classifier)

    elif architecture == "vgg11_bn":
        # VGG: Add adaptive pool + flatten + FC
        classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(feature_dim * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )
        model = nn.Sequential(model, classifier)

    return model


def build_model(
    architecture: str = "resnet18",
    num_classes: int = 5,
    input_channels: int = 4,
    pretrained: bool = True,
) -> nn.Module:
    """
    Build complete classification model.

    Combines:
    1. Load pre-trained backbone
    2. Adapt first conv for 4-channel input
    3. Replace classifier head for num_classes output

    Args:
        architecture: Model architecture name
        num_classes: Number of output classes
        input_channels: Number of input channels
        pretrained: Whether to use pre-trained weights

    Returns:
        Complete model ready for training

    Raises:
        ValueError: If input_channels != 4 (only 4-channel supported)
    """
    if input_channels != 4:
        raise ValueError(f"Only 4-channel input supported, got {input_channels}")

    # Build backbone
    backbone, feature_dim = build_backbone(architecture, pretrained=pretrained)

    # Adapt for 4-channel input
    backbone = adapt_first_conv_for_4ch(backbone, architecture)

    # Replace classifier head
    model = replace_classifier_head(backbone, architecture, feature_dim, num_classes)

    return model


class ChlorellaClassifier(nn.Module):
    """
    Complete classifier for holographic microscopy images.

    Wraps the model building process into a single nn.Module for easier use.
    """

    def __init__(
        self,
        architecture: str = "resnet18",
        num_classes: int = 5,
        input_channels: int = 4,
        pretrained: bool = True,
    ):
        """
        Initialize classifier.

        Args:
            architecture: Model architecture name
            num_classes: Number of output classes
            input_channels: Number of input channels
            pretrained: Whether to use pre-trained weights
        """
        super().__init__()
        self.architecture = architecture
        self.num_classes = num_classes
        self.input_channels = input_channels

        # Build model
        self.model = build_model(
            architecture=architecture,
            num_classes=num_classes,
            input_channels=input_channels,
            pretrained=pretrained,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, 4, H, W)

        Returns:
            Logits (B, num_classes)
        """
        return self.model(x)

    def get_backbone_params(self):
        """
        Get parameters for backbone (for discriminative fine-tuning).

        Returns:
            Generator of parameters
        """
        if self.architecture in ["resnet18", "vgg11_bn"]:
            # Backbone is first part of the sequential model
            return self.model[0].parameters()
        else:
            return self.model.parameters()

    def get_classifier_params(self):
        """
        Get parameters for classifier head (for discriminative fine-tuning).

        Returns:
            Generator of parameters
        """
        if self.architecture in ["resnet18", "vgg11_bn"]:
            # Classifier is second part of the sequential model
            return self.model[1].parameters()
        else:
            return self.model.parameters()
