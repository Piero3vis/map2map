import numpy as np
import os
from scipy.ndimage import zoom

def create_3d_test_data(num_samples, hr_size, scale_factor, save_dir):
    """Create mock 3D test data with low and high resolution pairs.
    
    Args:
        num_samples: Number of samples to generate
        hr_size: Size of high resolution data (e.g., 64 for 64x64x64)
        scale_factor: Factor between LR and HR (e.g., 8)
        save_dir: Directory to save the data
    """
    lr_size = hr_size // scale_factor
    os.makedirs(os.path.join(save_dir, 'test'), exist_ok=True)
    
    print(f"Generating test data:")
    print(f"HR size: {hr_size}x{hr_size}x{hr_size}")
    print(f"LR size: {lr_size}x{lr_size}x{lr_size}")
    print(f"Scale factor: {scale_factor}")
    
    # Physical volume size (same for both resolutions)
    volume_size = 100  # arbitrary units
    
    for i in range(num_samples):
        # Create a grid for HR
        hr_grid = np.linspace(-volume_size/2, volume_size/2, hr_size)
        X, Y, Z = np.meshgrid(hr_grid, hr_grid, hr_grid, indexing='ij')
        
        # Create a structured 3D field (e.g., multiple Gaussian blobs)
        centers = np.random.rand(3, 3) * volume_size - volume_size/2  # 3 random centers
        field = np.zeros((hr_size, hr_size, hr_size))
        
        for cx, cy, cz in centers:
            # Create Gaussian blob
            sigma = volume_size/10  # size of the blob relative to volume
            field += np.exp(-((X-cx)**2 + (Y-cy)**2 + (Z-cz)**2) / (2*sigma**2))
            
        # Add some noise to HR
        field += np.random.normal(0, 0.1, field.shape)
        
        # Normalize HR to [0, 1]
        field = (field - field.min()) / (field.max() - field.min())
        
        # Create high-resolution data
        hr_data = field[np.newaxis, ...]  # Add channel dimension
        
        # Create low-resolution data by downsampling
        lr_field = zoom(field, 1/scale_factor, order=1)
        
        # Add noise to LR
        lr_field += np.random.normal(0, 0.05, lr_field.shape)
        
        # Normalize LR to [0, 1]
        lr_field = (lr_field - lr_field.min()) / (lr_field.max() - lr_field.min())
        
        lr_data = lr_field[np.newaxis, ...]  # Add channel dimension
        
        # Save the data
        np.save(
            os.path.join(save_dir, 'test', f'lr_scalar-{i:03d}.npy'),
            lr_data.astype(np.float32)
        )
        np.save(
            os.path.join(save_dir, 'test', f'hr_scalar-{i:03d}.npy'),
            hr_data.astype(np.float32)
        )
        
        print(f"Generated sample {i+1}/{num_samples}")
        print(f"HR shape: {hr_data.shape}")
        print(f"LR shape: {lr_data.shape}")

def main():
    hr_size = 64  # High resolution size
    scale_factor = 2  # Factor between LR and HR
    test_samples = 5
    save_dir = '.'
    
    create_3d_test_data(test_samples, hr_size, scale_factor, save_dir)
    
    print("\nTest data generation complete!")
    print(f"Created {test_samples} test samples")
    print(f"LR data shape: (1, {hr_size//scale_factor}, {hr_size//scale_factor}, {hr_size//scale_factor})")
    print(f"HR data shape: (1, {hr_size}, {hr_size}, {hr_size})")
    print(f"Saved in directory: {save_dir}/test")

if __name__ == '__main__':
    main() 