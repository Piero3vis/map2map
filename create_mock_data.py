import os
import numpy as np
from scipy.ndimage import zoom

def create_3d_data(num_samples, lr_size, scale_factor, save_dir, prefix):
    """Create mock 3D data with low and high resolution pairs.
    
    Args:
        num_samples: Number of samples to generate
        lr_size: Size of low resolution data (lr_size x lr_size x lr_size)
        scale_factor: Factor between LR and HR (e.g., 2 means HR is 2x bigger)
        save_dir: Directory to save the data
        prefix: Prefix for filenames ('train' or 'val')
    """
    hr_size = lr_size * scale_factor
    os.makedirs(os.path.join(save_dir, prefix), exist_ok=True)
    
    print(f"Generating {prefix} data:")
    print(f"LR size: {lr_size}x{lr_size}x{lr_size}")
    print(f"HR size: {hr_size}x{hr_size}x{hr_size}")
    print(f"Scale factor: {scale_factor}")
    
    for i in range(num_samples):
        # Create a random 3D field with some structure (in HR)
        x = np.linspace(-4, 4, hr_size)
        y = np.linspace(-4, 4, hr_size)
        z = np.linspace(-4, 4, hr_size)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Create a structured 3D field (e.g., multiple Gaussian blobs)
        centers = np.random.rand(3, 3) * 8 - 4  # 3 random centers
        field = np.zeros((hr_size, hr_size, hr_size))
        
        for cx, cy, cz in centers:
            field += np.exp(-((X-cx)**2 + (Y-cy)**2 + (Z-cz)**2) / 2)
            
        # Add some noise to HR
        field += np.random.normal(0, 0.1, field.shape)
        
        # Normalize HR to [0, 1]
        field = (field - field.min()) / (field.max() - field.min())
        
        # Create high-resolution data
        hr_data = field[np.newaxis, ...]  # Add channel dimension
        
        # Create low-resolution data by downsampling
        lr_field = zoom(field, 1/scale_factor, order=1)  # order=1 for bilinear interpolation
        
        # Add noise to LR
        lr_field += np.random.normal(0, 0.1, lr_field.shape)
        
        # Normalize LR to [0, 1]
        lr_field = (lr_field - lr_field.min()) / (lr_field.max() - lr_field.min())
        
        lr_data = lr_field[np.newaxis, ...]  # Add channel dimension
        
        # Print shapes for the first sample
        if i == 0:
            print(f"\nSample shapes:")
            print(f"LR data shape: {lr_data.shape}")
            print(f"HR data shape: {hr_data.shape}")
        
        # Save the data
        np.save(
            os.path.join(save_dir, prefix, f'lr_scalar-{i:03d}.npy'),
            lr_data.astype(np.float32)
        )
        np.save(
            os.path.join(save_dir, prefix, f'hr_scalar-{i:03d}.npy'),
            hr_data.astype(np.float32)
        )

def main():
    lr_size = 32         # Start with LR size
    scale_factor = 2     # HR will be 2x the resolution of LR
    train_samples = 20   # Number of training samples
    val_samples = 5      # Number of validation samples
    save_dir = '.'
    
    # Create training data
    create_3d_data(train_samples, lr_size, scale_factor, save_dir, 'train')
    
    # Create validation data
    create_3d_data(val_samples, lr_size, scale_factor, save_dir, 'val')
    
    print("\nData generation complete!")
    print(f"Created {train_samples} training samples and {val_samples} validation samples")
    print(f"LR data shape: (1, {lr_size}, {lr_size}, {lr_size})")
    print(f"HR data shape: (1, {lr_size*scale_factor}, {lr_size*scale_factor}, {lr_size*scale_factor})")
    print(f"Scale factor: {scale_factor}x")
    print(f"Saved in directories: {save_dir}/train and {save_dir}/val")

if __name__ == '__main__':
    main()

