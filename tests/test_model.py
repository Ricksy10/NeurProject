"""
Unit tests for model architecture in neur/model.py
"""

import pytest

# These tests will be fully implemented after model.py is created


class TestModelAdaptation:
    """Tests for first conv adaptation (T018)."""

    @pytest.mark.skip(reason="Model implementation not yet complete")
    def test_adapt_first_conv_for_4ch(self):
        """
        Test that adapt_first_conv_for_4ch() converts conv1 from 3 to 4 channels.

        Should:
        - Convert conv1 from (64, 3, 7, 7) to (64, 4, 7, 7)
        - Preserve pre-trained weights for channels 0-2
        - Initialize channel 3 with small values or zeros
        """
        # TODO: Implement after model.py is created
        pass

    @pytest.mark.skip(reason="Model implementation not yet complete")
    def test_pretrained_weights_preserved(self):
        """
        Test that pre-trained weights are preserved during adaptation.
        """
        # TODO: Implement after model.py is created
        pass

    @pytest.mark.skip(reason="Model implementation not yet complete")
    def test_model_forward_pass(self):
        """
        Test that adapted model can perform forward pass with 4-channel input.
        """
        # TODO: Implement after model.py is created
        pass


class TestModelBuilder:
    """Tests for model building functions."""

    @pytest.mark.skip(reason="Model implementation not yet complete")
    def test_build_resnet18(self):
        """Test building ResNet18 backbone."""
        # TODO: Implement after model.py is created
        pass

    @pytest.mark.skip(reason="Model implementation not yet complete")
    def test_build_vgg11_bn(self):
        """Test building VGG11-BN backbone."""
        # TODO: Implement after model.py is created
        pass

    @pytest.mark.skip(reason="Model implementation not yet complete")
    def test_replace_classifier_head(self):
        """Test replacing classifier head for 5 classes."""
        # TODO: Implement after model.py is created
        pass
