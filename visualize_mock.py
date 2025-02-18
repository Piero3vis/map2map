import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
import glob
import os

def load_and_plot_pair(lr_path, hr_path, row, fig):
    """Load and plot a pair of LR and HR data."""
    # Load data
    lr_data = np.load(lr_path)[0]
    hr_data = np.load(hr_path)[0]
    
    # Create scaled coordinate grids (both from 0 to 64)
    lr_grid = np.linspace(0, 64, lr_data.shape[0])
    hr_grid = np.linspace(0, 64, hr_data.shape[0])
    
    lr_X, lr_Y, lr_Z = np.meshgrid(lr_grid, lr_grid, lr_grid)
    hr_X, hr_Y, hr_Z = np.meshgrid(hr_grid, hr_grid, hr_grid)
    
    # Low-resolution 3D plot
    fig.add_trace(go.Volume(
        x=lr_X.flatten(),
        y=lr_Y.flatten(),
        z=lr_Z.flatten(),
        value=lr_data.flatten(),
        opacity=0.3,
        surface_count=15,
        colorscale='viridis',
        showscale=True,
    ), row=row, col=1)
    
    # High-resolution 3D plot
    fig.add_trace(go.Volume(
        x=hr_X.flatten(),
        y=hr_Y.flatten(),
        z=hr_Z.flatten(),
        value=hr_data.flatten(),
        opacity=0.3,
        surface_count=15,
        colorscale='viridis',
        showscale=True,
    ), row=row, col=2)

# Create subplots (3 rows for train/val/test, 2 columns for LR/HR)
fig = make_subplots(
    rows=3, cols=2,
    specs=[[{'type': 'volume'}, {'type': 'volume'}],
           [{'type': 'volume'}, {'type': 'volume'}],
           [{'type': 'volume'}, {'type': 'volume'}]],
    subplot_titles=("Training LR", "Training HR",
                   "Validation LR", "Validation HR",
                   "Test LR", "Test HR")
)

# Get all sample numbers from training set
train_lr_files = sorted(glob.glob('train/lr_scalar-*.npy'))
sample_numbers = [int(f.split('-')[-1].split('.')[0]) for f in train_lr_files]

if sample_numbers:
    # Randomly select one sample number
    sample_num = random.choice(sample_numbers)
    
    # Format the sample number with leading zeros
    sample_str = f"{sample_num:03d}"
    sample_str = "003"
    
    # Load corresponding pairs from each set
    if os.path.exists(f'train/lr_scalar-{sample_str}.npy'):
        load_and_plot_pair(
            f'train/lr_scalar-{sample_str}.npy',
            f'train/hr_scalar-{sample_str}.npy',
            1, fig
        )
    
    if os.path.exists(f'val/lr_scalar-{sample_str}.npy'):
        load_and_plot_pair(
            f'val/lr_scalar-{sample_str}.npy',
            f'val/hr_scalar-{sample_str}.npy',
            2, fig
        )
    
    if os.path.exists(f'test/lr_scalar-{sample_str}.npy'):
        load_and_plot_pair(
            f'test/lr_scalar-{sample_str}.npy',
            f'test/hr_scalar-{sample_str}.npy',
            3, fig
        )
    
    # Layout adjustments
    fig.update_layout(
        title_text=f"3D Gaussian Blob Comparison (Sample #{sample_str})",
        height=1800,
        width=1200,
    )
    
    # Set all plots to range 0-64 and cube aspect ratio
    for i in range(1, 4):  # For each row
        fig.update_layout({
            f'scene{i if i > 1 else ""}': dict(
                aspectmode='cube',
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                xaxis=dict(range=[0, 64]),
                yaxis=dict(range=[0, 64]),
                zaxis=dict(range=[0, 64])
            ),
            f'scene{i*2 if i > 1 else "2"}': dict(
                aspectmode='cube',
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                xaxis=dict(range=[0, 64]),
                yaxis=dict(range=[0, 64]),
                zaxis=dict(range=[0, 64])
            )
        })
    
    fig.show()
else:
    print("No training samples found!")
