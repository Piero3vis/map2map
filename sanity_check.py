import numpy as np
import torch
import os
from map2map.models.srsgan import G, D  # Import SRSGAN models instead of UNet

def load_sample_data():
    """Load one sample from training data"""
    train_file_lr = os.path.join('train', 'lr_scalar-000.npy')
    train_file_hr = os.path.join('train', 'hr_scalar-000.npy')
    
    if not os.path.exists(train_file_lr) or not os.path.exists(train_file_hr):
        raise FileNotFoundError("Training data not found. Run create_mock_data.py first.")
    
    lr_data = np.load(train_file_lr)
    hr_data = np.load(train_file_hr)
    
    print("Data shapes:")
    print(f"LR data: {lr_data.shape}")
    print(f"HR data: {hr_data.shape}")
    
    return lr_data, hr_data

class DimensionLogger(torch.nn.Module):
    """Wrapper to log dimensions at each step"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        def hook_fn(name):
            def fn(module, input, output):
                if isinstance(input, tuple):
                    input = input[0]
                print(f"\n{name}:")
                print(f"Input shape: {input.shape}")
                print(f"Output shape: {output.shape}")
            return fn
        
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Conv3d, torch.nn.ConvTranspose3d)):
                self.hooks.append(
                    module.register_forward_hook(hook_fn(name))
                )
    
    def forward(self, x):
        return self.model(x)

def test_network():
    """Test SRSGAN configuration"""
    # Load sample data
    lr_data, hr_data = load_sample_data()
    
    # Convert to torch tensors
    lr_tensor = torch.from_numpy(lr_data).float()
    hr_tensor = torch.from_numpy(hr_data).float()
    batch_lr = lr_tensor.unsqueeze(0)
    batch_hr = hr_tensor.unsqueeze(0)
    
    print(f"\nInput tensor shapes:")
    print(f"LR (with batch): {batch_lr.shape}")
    print(f"HR (with batch): {batch_hr.shape}")
    
    # Generator configuration (from mock-train.sh)
    g_config = {
        "chan_base": 64,
        "chan_min": 32,
        "chan_max": 256,
        "cat_noise": True,
        "scale_factor": 2  # For 32->64 upscaling
    }
    
    print("\nGenerator configuration:")
    for key, value in g_config.items():
        print(f"{key}: {value}")
    
    # Create Generator
    print("\nCreating Generator...")
    generator = G(in_chan=1, out_chan=1, **g_config)
    generator.eval()
    
    # Create Discriminator
    print("\nCreating Discriminator...")
    discriminator = D(in_chan=1, out_chan=1)  # Added out_chan parameter
    discriminator.eval()
    
    # Test forward passes
    with torch.no_grad():
        try:
            print("\nTesting Generator forward pass...")
            generator_with_hooks = DimensionLogger(generator)
            sr_output = generator_with_hooks(batch_lr)
            print(f"\nGenerator output shape: {sr_output.shape}")
            
            print("\nTesting Discriminator forward pass...")
            discriminator_with_hooks = DimensionLogger(discriminator)
            d_real = discriminator_with_hooks(batch_hr)  # Test with HR
            d_fake = discriminator_with_hooks(sr_output)  # Test with SR
            print(f"\nDiscriminator output shapes:")
            print(f"Real: {d_real.shape}")
            print(f"Fake: {d_fake.shape}")
            
            # Verify shapes
            assert sr_output.shape == batch_hr.shape, \
                f"Generator output shape {sr_output.shape} doesn't match HR shape {batch_hr.shape}"
            
            print("\nSanity check passed! Both Generator and Discriminator are working correctly.")
            
        except Exception as e:
            print(f"\nError during forward pass: {str(e)}")
            print("Traceback:")
            import traceback
            traceback.print_exc()
            print("\nSanity check failed!")
        
        finally:
            # Clean up hooks
            if 'generator_with_hooks' in locals():
                for hook in generator_with_hooks.hooks:
                    hook.remove()
            if 'discriminator_with_hooks' in locals():
                for hook in discriminator_with_hooks.hooks:
                    hook.remove()

if __name__ == "__main__":
    test_network() 