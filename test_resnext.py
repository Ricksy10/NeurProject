"""Quick test script to verify ResNeXt-50 integration."""

import torch
from neur.model import ChlorellaClassifier, build_model

print("Testing ResNeXt-50 (32x4d) integration...")
print("=" * 60)

# Test 1: Build model via ChlorellaClassifier
print("\n1. Testing ChlorellaClassifier with ResNeXt-50...")
model = ChlorellaClassifier(architecture="resnext50_32x4d", pretrained=False)
print(f"   ✓ Model architecture: {model.architecture}")
print(f"   ✓ Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test 2: Forward pass
print("\n2. Testing forward pass...")
batch_size = 4
x = torch.randn(batch_size, 4, 224, 224)
out = model(x)
print(f"   ✓ Input shape: {x.shape}")
print(f"   ✓ Output shape: {out.shape}")
assert out.shape == (batch_size, 5), f"Expected (4, 5), got {out.shape}"

# Test 3: Discriminative parameter groups
print("\n3. Testing discriminative fine-tuning parameter groups...")
backbone_params = list(model.get_backbone_params())
classifier_params = list(model.get_classifier_params())
print(f"   ✓ Backbone parameters: {len(backbone_params)} groups")
print(f"   ✓ Classifier parameters: {len(classifier_params)} groups")

# Test 4: Compare with ResNet18
print("\n4. Comparing architectures...")
resnet18 = ChlorellaClassifier(architecture="resnet18", pretrained=False)
resnext50 = ChlorellaClassifier(architecture="resnext50_32x4d", pretrained=False)
vgg11 = ChlorellaClassifier(architecture="vgg11_bn", pretrained=False)

resnet18_params = sum(p.numel() for p in resnet18.parameters())
resnext50_params = sum(p.numel() for p in resnext50.parameters())
vgg11_params = sum(p.numel() for p in vgg11.parameters())

print(f"   ResNet18:      {resnet18_params:,} parameters")
print(f"   ResNeXt-50:    {resnext50_params:,} parameters")
print(f"   VGG11-BN:      {vgg11_params:,} parameters")
print(f"   ResNeXt-50 is {resnext50_params / resnet18_params:.1f}x larger than ResNet18")

# Test 5: Verify 4-channel input
print("\n5. Testing 4-channel input adaptation...")
model_direct = build_model(architecture="resnext50_32x4d", pretrained=False)
# Check first conv layer
first_conv = model_direct[0][0]  # Sequential[backbone][conv1]
print(f"   ✓ First conv input channels: {first_conv.in_channels}")
print(f"   ✓ First conv output channels: {first_conv.out_channels}")
assert first_conv.in_channels == 4, f"Expected 4 input channels, got {first_conv.in_channels}"

print("\n" + "=" * 60)
print("✓ All ResNeXt-50 integration tests passed!")
print("\nResNeXt-50 is now available for training:")
print("  python scripts/train.py --model-name resnext50_32x4d")
