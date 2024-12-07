import torch
import torch.nn as nn
from model import Net


def test_parameter_count():
    """Test that model has less than 20k parameters"""
    model = Net()
    total_params = sum(p.numel() for p in model.parameters())
    assert (
        total_params < 20000
    ), f"Model has {total_params} parameters, should be less than 20000"


def test_no_batch_norm():
    """Test that model does not use batch normalization"""
    model = Net()
    has_batchnorm = any(isinstance(m, nn.BatchNorm2d) for m in model.modules())
    assert has_batchnorm, "Model should not use batch normalization"


def test_no_dropout():
    """Test that model does not use dropout"""
    model = Net()
    has_dropout = any(isinstance(m, nn.Dropout) for m in model.modules())
    assert has_dropout, "Model should use dropout"


def test_gap_or_fc():
    """Test that model uses GAP (using AdaptiveAvgPool2d) instead of FC layer"""
    model = Net()
    # Check for presence of AdaptiveAvgPool2d
    has_gap = any(
        isinstance(m, (nn.AvgPool2d, nn.AdaptiveAvgPool2d)) for m in model.modules()
    )
    # Check that there's no Linear layer before the final classification layer
    modules = list(model.modules())
    linear_count = sum(1 for m in modules[:-1] if isinstance(m, nn.Linear))

    assert has_gap, "Model should use Global Average Pooling (AdaptiveAvgPool2d)"
    assert linear_count == 0, "Model should not use intermediate Fully Connected layers"
