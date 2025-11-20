"""
Model architecture and utilities for the Chlorella classification pipeline.

Handles:
- Pre-trained backbone loading (ResNet18, VGG11-BN, ResNeXt-50)
- First conv layer adaptation for 4-channel input
- Classifier head replacement for 5 classes
- Model building utilities
"""

import ssl
import warnings
from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models

# Suppress deprecation warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')
warnings.filterwarnings('ignore', message='.*pretrained.*')
warnings.filterwarnings('ignore', message='.*weights.*')

# Fix SSL certificate verification issue on macOS
# This allows downloading pre-trained weights from PyTorch hub
ssl._create_default_https_context = ssl._create_unverified_context


def build_backbone(
    architecture: str = "resnet18", pretrained: bool = True
) -> Tuple[nn.Module, int]:
    """
    Load pre-trained backbone model.

    Args:
        architecture: Model architecture name
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

    elif architecture == "resnext50_32x4d":
        # ResNeXt-50 (32x4d) with grouped convolutions for better feature learning
        # Research shows 98.45% accuracy on algae classification
        model = models.resnext50_32x4d(pretrained=pretrained)
        feature_dim = model.fc.in_features  # 2048 for ResNeXt-50
        # Remove final FC layer (will replace with custom head)
        model = nn.Sequential(*list(model.children())[:-1])
        return model, feature_dim

    elif architecture == "vgg11_bn":
        model = models.vgg11_bn(pretrained=pretrained)
        # VGG has two parts: features and classifier
        feature_extractor = model.features
        feature_dim = 512  # VGG11 outputs 512 features
        return feature_extractor, feature_dim
    
    elif architecture == "efficientnet_b0":
        # EfficientNet-B0: Excellent accuracy with low parameters
        # Great for medical/microscopy images
        model = models.efficientnet_b0(pretrained=pretrained)
        feature_dim = model.classifier[1].in_features  # 1280 for B0
        # Use only the features part (before avgpool and classifier)
        model = model.features
        return model, feature_dim
    
    elif architecture == "efficientnet_b1":
        # EfficientNet-B1: Better accuracy, still efficient
        model = models.efficientnet_b1(pretrained=pretrained)
        feature_dim = model.classifier[1].in_features  # 1280 for B1
        # Use only the features part
        model = model.features
        return model, feature_dim
    
    elif architecture == "efficientnet_b3":
        # EfficientNet-B3: High accuracy for challenging tasks
        model = models.efficientnet_b3(pretrained=pretrained)
        feature_dim = model.classifier[1].in_features  # 1536 for B3
        # Use only the features part
        model = model.features
        return model, feature_dim

    else:
        raise ValueError(
            f"Unsupported architecture: {architecture}. "
            f"Supported: 'resnet18', 'resnext50_32x4d', 'vgg11_bn', "
            f"'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b3'"
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
    if architecture in ["resnet18", "resnext50_32x4d"]:
        # For ResNet/ResNeXt wrapped in Sequential, first conv is at model[0]
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
    
    elif architecture in ["efficientnet_b0", "efficientnet_b1", "efficientnet_b3"]:
        # EfficientNet: modify first conv in model[0][0] (since model = model.features)
        old_conv = model[0][0]
        
        new_conv = nn.Conv2d(
            in_channels=4,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_conv.weight
            new_conv.weight[:, 3:, :, :] = torch.randn_like(new_conv.weight[:, 3:, :, :]) * 0.01
            
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)
        
        model[0][0] = new_conv

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
    if architecture in ["resnet18", "resnext50_32x4d"]:
        # ResNet/ResNeXt: Add global average pool + FC
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
    
    elif architecture in ["efficientnet_b0", "efficientnet_b1", "efficientnet_b3"]:
        # EfficientNet: Add dropout + FC classifier
        classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, num_classes)
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
        if self.architecture in ["resnet18", "resnext50_32x4d", "vgg11_bn", 
                                  "efficientnet_b0", "efficientnet_b1", "efficientnet_b3"]:
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
        if self.architecture in ["resnet18", "resnext50_32x4d", "vgg11_bn",
                                  "efficientnet_b0", "efficientnet_b1", "efficientnet_b3"]:
            # Classifier is second part of the sequential model
            return self.model[1].parameters()
        else:
            return self.model.parameters()
