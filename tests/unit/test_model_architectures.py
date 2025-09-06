"""
Test suite for model architectures (Conv4, ResNet12, WideResNet, MetaModule).

Tests the new model architectures implemented in src/meta_learning/models/conv4.py:
- Conv4 backbone functionality and configuration
- ResNet12 architecture with residual connections
- WideResNet implementation with configurable width/depth
- MetaModule components for gradient-based meta-learning
"""

import pytest
import torch
import torch.nn as nn
import math
from unittest.mock import patch

from meta_learning.models.conv4 import (
    Conv4, ResNet12, WideResNet, WideBasicBlock,
    MetaModule, MetaLinear, MetaConv2d
)


class TestConv4:
    """Test Conv4 backbone architecture."""
    
    def test_conv4_initialization(self):
        """Test Conv4 initialization with different configurations."""
        # Default configuration
        model = Conv4()
        assert model is not None
        
        # Custom output dimension
        model = Conv4(out_dim=128, p_drop=0.1, input_channels=1)
        assert hasattr(model, 'features')
        assert hasattr(model, 'dropout')
        assert hasattr(model, 'projection')
        
    def test_conv4_forward_pass(self):
        """Test Conv4 forward pass with different input sizes."""
        model = Conv4(out_dim=64)
        
        # Standard 84x84 input (MiniImageNet size)
        x = torch.randn(5, 3, 84, 84)
        output = model(x)
        assert output.shape == (5, 64)
        
        # Different input size (28x28 for Omniglot)
        x = torch.randn(3, 3, 28, 28)
        output = model(x)
        assert output.shape == (3, 64)
        
        # Grayscale input
        model = Conv4(input_channels=1)
        x = torch.randn(4, 1, 84, 84)
        output = model(x)
        assert output.shape == (4, 64)
        
    def test_conv4_projection_layer(self):
        """Test Conv4 with custom output projection."""
        # With projection
        model = Conv4(out_dim=256)
        assert not isinstance(model.projection, nn.Identity)
        
        x = torch.randn(2, 3, 84, 84)
        output = model(x)
        assert output.shape == (2, 256)
        
        # Without projection (default 64-dim)
        model = Conv4(out_dim=64)
        assert isinstance(model.projection, nn.Identity)
        
    def test_conv4_dropout(self):
        """Test Conv4 dropout functionality."""
        # With dropout
        model = Conv4(p_drop=0.5)
        assert not isinstance(model.dropout, nn.Identity)
        
        # Without dropout
        model = Conv4(p_drop=0.0)
        assert isinstance(model.dropout, nn.Identity)


class TestResNet12:
    """Test ResNet12 architecture."""
    
    def test_resnet12_initialization(self):
        """Test ResNet12 initialization."""
        model = ResNet12()
        assert hasattr(model, 'block1')
        assert hasattr(model, 'block2')
        assert hasattr(model, 'block3')
        assert hasattr(model, 'block4')
        assert hasattr(model, 'global_pool')
        assert hasattr(model, 'projection')
        
    def test_resnet12_forward_pass(self):
        """Test ResNet12 forward pass."""
        model = ResNet12(out_dim=512)
        
        # Standard input
        x = torch.randn(4, 3, 84, 84)
        output = model(x)
        assert output.shape == (4, 512)
        
        # Custom output dimension
        model = ResNet12(out_dim=256)
        output = model(x)
        assert output.shape == (4, 256)
        
    def test_resnet12_residual_connection(self):
        """Test ResNet12 residual connections."""
        model = ResNet12()
        
        # Check that shortcut connections exist
        # This is a structural test - in practice would check gradient flow
        x = torch.randn(2, 3, 84, 84)
        
        # Enable gradient computation
        x.requires_grad_(True)
        output = model(x)
        
        # Compute gradients to verify residual connections help gradient flow
        loss = output.sum()
        loss.backward()
        
        # Input should have gradients (indicating successful backprop)
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))
        
    def test_resnet12_different_inputs(self):
        """Test ResNet12 with different input configurations."""
        # Different input channels
        model = ResNet12(input_channels=1, out_dim=256)
        x = torch.randn(3, 1, 84, 84)
        output = model(x)
        assert output.shape == (3, 256)
        
        # Different dropout rate
        model = ResNet12(dropout_rate=0.2)
        x = torch.randn(2, 3, 84, 84)
        output = model(x)
        assert output.shape == (2, 512)  # Default out_dim


class TestWideResNet:
    """Test WideResNet architecture."""
    
    def test_wideresnet_initialization(self):
        """Test WideResNet initialization with different configurations."""
        # Default configuration
        model = WideResNet()
        assert hasattr(model, 'conv1')
        assert hasattr(model, 'layer1')
        assert hasattr(model, 'layer2')
        assert hasattr(model, 'layer3')
        
        # Custom configuration
        model = WideResNet(depth=16, widen_factor=8, out_dim=512)
        assert model is not None
        
    def test_wideresnet_depth_validation(self):
        """Test WideResNet depth validation."""
        # Valid depth (6n+4 format)
        model = WideResNet(depth=28)  # 28 = 6*4 + 4
        assert model is not None
        
        # Invalid depth should raise assertion error
        with pytest.raises(AssertionError):
            WideResNet(depth=25)  # Not 6n+4 format
            
    def test_wideresnet_forward_pass(self):
        """Test WideResNet forward pass."""
        model = WideResNet(depth=16, widen_factor=4, out_dim=256)
        
        x = torch.randn(3, 3, 84, 84)
        output = model(x)
        assert output.shape == (3, 256)
        
    def test_wideresnet_widening_factor(self):
        """Test different widening factors."""
        # Narrow network
        model_narrow = WideResNet(depth=16, widen_factor=1)
        
        # Wide network  
        model_wide = WideResNet(depth=16, widen_factor=10)
        
        # Wide network should have more parameters
        narrow_params = sum(p.numel() for p in model_narrow.parameters())
        wide_params = sum(p.numel() for p in model_wide.parameters())
        
        assert wide_params > narrow_params
        
    def test_wideresnet_different_inputs(self):
        """Test WideResNet with different input configurations."""
        model = WideResNet(input_channels=1, dropout_rate=0.1)
        
        x = torch.randn(2, 1, 28, 28)
        output = model(x)
        assert len(output.shape) == 2  # Should be flattened
        assert output.shape[0] == 2


class TestWideBasicBlock:
    """Test WideBasicBlock component."""
    
    def test_wide_basic_block_initialization(self):
        """Test WideBasicBlock initialization."""
        block = WideBasicBlock(in_planes=16, planes=32, dropout_rate=0.1)
        
        assert hasattr(block, 'bn1')
        assert hasattr(block, 'conv1')
        assert hasattr(block, 'dropout')
        assert hasattr(block, 'bn2')
        assert hasattr(block, 'conv2')
        assert hasattr(block, 'shortcut')
        
    def test_wide_basic_block_forward(self):
        """Test WideBasicBlock forward pass."""
        block = WideBasicBlock(in_planes=16, planes=16, dropout_rate=0.1)
        
        x = torch.randn(2, 16, 32, 32)
        output = block(x)
        assert output.shape == x.shape  # Same dimensions with matching planes
        
    def test_wide_basic_block_with_stride(self):
        """Test WideBasicBlock with stride (downsampling)."""
        block = WideBasicBlock(in_planes=16, planes=32, dropout_rate=0.1, stride=2)
        
        x = torch.randn(2, 16, 32, 32)
        output = block(x)
        
        # Should downsample spatial dimensions and change channels
        assert output.shape == (2, 32, 16, 16)
        
    def test_wide_basic_block_shortcut(self):
        """Test shortcut connection in WideBasicBlock."""
        # When input/output planes differ, should have conv shortcut
        block = WideBasicBlock(in_planes=16, planes=32, dropout_rate=0.1)
        assert len(block.shortcut) > 0  # Should have conv layer
        
        # When input/output planes match, should have identity shortcut
        block = WideBasicBlock(in_planes=32, planes=32, dropout_rate=0.1)
        assert len(block.shortcut) == 0  # Should be empty (identity)


class TestMetaModule:
    """Test MetaModule base class for gradient-based meta-learning."""
    
    def test_meta_module_initialization(self):
        """Test MetaModule initialization."""
        module = MetaModule()
        
        assert hasattr(module, 'meta_named_parameters')
        assert hasattr(module, 'meta_parameters')
        
    def test_meta_named_parameters(self):
        """Test meta_named_parameters generator."""
        # Create a simple meta module with parameters
        class SimpleMetaModule(MetaModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
                
        module = SimpleMetaModule()
        
        # Should be able to iterate through named parameters
        param_names = list(name for name, _ in module.meta_named_parameters())
        assert 'linear.weight' in param_names
        assert 'linear.bias' in param_names
        
    def test_meta_parameters(self):
        """Test meta_parameters generator."""
        class SimpleMetaModule(MetaModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
                
        module = SimpleMetaModule()
        
        # Should be able to iterate through parameters
        params = list(module.meta_parameters())
        assert len(params) == 2  # weight and bias
        assert all(isinstance(p, torch.Tensor) for p in params)


class TestMetaLinear:
    """Test MetaLinear layer for meta-learning."""
    
    def test_meta_linear_initialization(self):
        """Test MetaLinear initialization."""
        layer = MetaLinear(10, 5)
        
        assert layer.in_features == 10
        assert layer.out_features == 5
        assert hasattr(layer, 'weight')
        assert hasattr(layer, 'bias')
        
        # Test without bias
        layer_no_bias = MetaLinear(10, 5, bias=False)
        assert layer_no_bias.bias is None
        
    def test_meta_linear_parameter_initialization(self):
        """Test MetaLinear parameter initialization."""
        layer = MetaLinear(10, 5)
        
        # Check weight initialization (should follow Kaiming uniform)
        assert layer.weight.shape == (5, 10)
        assert not torch.allclose(layer.weight, torch.zeros_like(layer.weight))
        
        # Check bias initialization
        if layer.bias is not None:
            assert layer.bias.shape == (5,)
            # Bias should be initialized within reasonable bounds
            fan_in = layer.weight.shape[1]
            bound = 1 / math.sqrt(fan_in)
            assert torch.all(torch.abs(layer.bias) <= bound + 1e-6)
            
    def test_meta_linear_forward_without_params(self):
        """Test MetaLinear forward pass without external parameters."""
        layer = MetaLinear(10, 5)
        x = torch.randn(3, 10)
        
        output = layer(x)
        assert output.shape == (3, 5)
        
        # Should be equivalent to regular linear layer
        regular_layer = nn.Linear(10, 5)
        regular_layer.weight.data = layer.weight.data.clone()
        regular_layer.bias.data = layer.bias.data.clone()
        
        regular_output = regular_layer(x)
        assert torch.allclose(output, regular_output, atol=1e-6)
        
    def test_meta_linear_forward_with_params(self):
        """Test MetaLinear forward pass with external parameters."""
        layer = MetaLinear(10, 5)
        x = torch.randn(3, 10)
        
        # Custom parameters
        custom_weight = torch.randn(5, 10)
        custom_bias = torch.randn(5)
        params = {'weight': custom_weight, 'bias': custom_bias}
        
        output = layer(x, params=params)
        assert output.shape == (3, 5)
        
        # Should use custom parameters, not layer's own parameters
        expected_output = torch.nn.functional.linear(x, custom_weight, custom_bias)
        assert torch.allclose(output, expected_output, atol=1e-6)
        
    def test_meta_linear_partial_params(self):
        """Test MetaLinear with partial parameter override."""
        layer = MetaLinear(10, 5)
        x = torch.randn(3, 10)
        
        # Only override weight
        custom_weight = torch.randn(5, 10)
        params = {'weight': custom_weight}
        
        output = layer(x, params=params)
        
        # Should use custom weight but original bias
        expected_output = torch.nn.functional.linear(x, custom_weight, layer.bias)
        assert torch.allclose(output, expected_output, atol=1e-6)


class TestMetaConv2d:
    """Test MetaConv2d layer for meta-learning."""
    
    def test_meta_conv2d_initialization(self):
        """Test MetaConv2d initialization."""
        layer = MetaConv2d(3, 32, 3, stride=1, padding=1)
        
        assert layer.in_channels == 3
        assert layer.out_channels == 32
        assert layer.kernel_size == 3
        assert layer.stride == 1
        assert layer.padding == 1
        assert hasattr(layer, 'weight')
        assert hasattr(layer, 'bias')
        
        # Test without bias
        layer_no_bias = MetaConv2d(3, 32, 3, bias=False)
        assert layer_no_bias.bias is None
        
    def test_meta_conv2d_parameter_initialization(self):
        """Test MetaConv2d parameter initialization."""
        layer = MetaConv2d(3, 32, 3)
        
        # Check weight shape
        assert layer.weight.shape == (32, 3, 3, 3)
        assert not torch.allclose(layer.weight, torch.zeros_like(layer.weight))
        
        # Check bias initialization
        if layer.bias is not None:
            assert layer.bias.shape == (32,)
            
    def test_meta_conv2d_forward_without_params(self):
        """Test MetaConv2d forward pass without external parameters."""
        layer = MetaConv2d(3, 32, 3, padding=1)
        x = torch.randn(2, 3, 28, 28)
        
        output = layer(x)
        assert output.shape == (2, 32, 28, 28)
        
    def test_meta_conv2d_forward_with_params(self):
        """Test MetaConv2d forward pass with external parameters."""
        layer = MetaConv2d(3, 32, 3, padding=1)
        x = torch.randn(2, 3, 28, 28)
        
        # Custom parameters
        custom_weight = torch.randn(32, 3, 3, 3)
        custom_bias = torch.randn(32)
        params = {'weight': custom_weight, 'bias': custom_bias}
        
        output = layer(x, params=params)
        assert output.shape == (2, 32, 28, 28)
        
        # Should use custom parameters
        expected_output = torch.nn.functional.conv2d(
            x, custom_weight, custom_bias, layer.stride, layer.padding
        )
        assert torch.allclose(output, expected_output, atol=1e-6)
        
    def test_meta_conv2d_different_configurations(self):
        """Test MetaConv2d with different kernel sizes and strides."""
        # 5x5 kernel with stride 2
        layer = MetaConv2d(3, 64, 5, stride=2, padding=2)
        x = torch.randn(1, 3, 84, 84)
        
        output = layer(x)
        assert output.shape == (1, 64, 42, 42)  # Downsampled by stride 2
        
        # 1x1 kernel (pointwise convolution)
        layer = MetaConv2d(32, 16, 1)
        x = torch.randn(2, 32, 14, 14)
        
        output = layer(x)
        assert output.shape == (2, 16, 14, 14)


class TestModelIntegration:
    """Integration tests for model architectures."""
    
    def test_models_on_few_shot_task(self):
        """Test all models on a simulated few-shot learning task."""
        models = [
            Conv4(out_dim=64),
            ResNet12(out_dim=64),
            WideResNet(depth=16, widen_factor=2, out_dim=64)
        ]
        
        # Simulate 5-way 5-shot support set
        support_x = torch.randn(25, 3, 84, 84)  # 5 classes × 5 shots
        query_x = torch.randn(15, 3, 84, 84)    # 5 classes × 3 queries
        
        for model in models:
            # Extract features
            support_features = model(support_x)
            query_features = model(query_x)
            
            assert support_features.shape == (25, 64)
            assert query_features.shape == (15, 64)
            
            # Features should be different (not all zeros/ones)
            assert not torch.allclose(support_features, torch.zeros_like(support_features))
            assert not torch.allclose(query_features, torch.zeros_like(query_features))
            
    def test_model_gradient_flow(self):
        """Test that gradients flow properly through all models."""
        models = [
            Conv4(out_dim=32),
            ResNet12(out_dim=32),
            WideResNet(depth=16, widen_factor=1, out_dim=32)
        ]
        
        x = torch.randn(4, 3, 84, 84, requires_grad=True)
        
        for model in models:
            output = model(x)
            loss = output.sum()
            loss.backward()
            
            # Check that input has gradients
            assert x.grad is not None
            assert not torch.allclose(x.grad, torch.zeros_like(x.grad))
            
            # Check that model parameters have gradients
            for param in model.parameters():
                if param.requires_grad:
                    assert param.grad is not None
                    
            # Reset gradients for next model
            x.grad = None
            model.zero_grad()
            
    def test_meta_modules_composition(self):
        """Test composing MetaLinear and MetaConv2d layers."""
        class SimpleMetaNet(MetaModule):
            def __init__(self):
                super().__init__()
                self.conv = MetaConv2d(3, 16, 3, padding=1)
                self.linear = MetaLinear(16 * 28 * 28, 10)
                
            def forward(self, x, params=None):
                conv_params = params.get('conv', {}) if params else {}
                linear_params = params.get('linear', {}) if params else {}
                
                x = self.conv(x, conv_params)
                x = torch.relu(x)
                x = torch.adaptive_avg_pool2d(x, (28, 28))
                x = x.view(x.size(0), -1)
                x = self.linear(x, linear_params)
                return x
                
        model = SimpleMetaNet()
        x = torch.randn(2, 3, 84, 84)
        
        # Forward without parameters
        output1 = model(x)
        assert output1.shape == (2, 10)
        
        # Forward with custom parameters
        params = {
            'conv': {'weight': torch.randn(16, 3, 3, 3)},
            'linear': {'weight': torch.randn(10, 16 * 28 * 28)}
        }
        output2 = model(x, params)
        assert output2.shape == (2, 10)
        
        # Outputs should be different with different parameters
        assert not torch.allclose(output1, output2, atol=1e-3)


@pytest.fixture
def sample_images():
    """Fixture providing sample images for testing."""
    return {
        'omniglot': torch.randn(5, 1, 28, 28),
        'miniimagenet': torch.randn(5, 3, 84, 84),
        'cifar': torch.randn(5, 3, 32, 32)
    }


class TestModelCompatibility:
    """Test model compatibility with different datasets and scenarios."""
    
    def test_models_with_different_datasets(self, sample_images):
        """Test models with different dataset formats."""
        for dataset_name, images in sample_images.items():
            channels = images.shape[1]
            
            # Test Conv4
            if channels == 1:
                model = Conv4(input_channels=1, out_dim=64)
            else:
                model = Conv4(input_channels=3, out_dim=64)
                
            output = model(images)
            assert output.shape == (5, 64)
            
            # Test ResNet12
            if channels == 1:
                model = ResNet12(input_channels=1, out_dim=128)
            else:
                model = ResNet12(input_channels=3, out_dim=128)
                
            output = model(images)
            assert output.shape == (5, 128)
            
    def test_model_memory_efficiency(self):
        """Test that models don't use excessive memory."""
        # Test with reasonably large batch
        x = torch.randn(32, 3, 84, 84)
        
        models = [
            Conv4(out_dim=64),
            ResNet12(out_dim=64),
            WideResNet(depth=16, widen_factor=2, out_dim=64)
        ]
        
        for model in models:
            # Should not raise memory errors
            output = model(x)
            assert output.shape == (32, 64)
            
            # Clean up
            del output
            
    def test_model_determinism(self):
        """Test that models produce deterministic outputs given same seed."""
        torch.manual_seed(42)
        model1 = Conv4(out_dim=64)
        x = torch.randn(4, 3, 84, 84)
        
        torch.manual_seed(42)  # Reset seed
        output1 = model1(x)
        
        torch.manual_seed(42)  # Reset again
        output2 = model1(x)
        
        # Should produce identical outputs
        assert torch.allclose(output1, output2, atol=1e-6)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])