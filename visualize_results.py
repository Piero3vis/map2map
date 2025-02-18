import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import glob
import os

def load_and_plot_triplet(lr_path, hr_path, sr_path, fig):
    """Load and plot a triplet of LR, HR, and SR data."""
    # Load data
    lr_data = np.load(lr_path)[0]  # Remove batch and channel dimensions
    hr_data = np.load(hr_path)[0]
    sr_data = np.load(sr_path)[0][0]
    
    print("Data shapes:")
    print(f"LR shape: {lr_data.shape}")
    print(f"HR shape: {hr_data.shape}")
    print(f"SR shape: {sr_data.shape}")
    
    # Create scaled coordinate grids (all from 0 to 64)
    lr_grid = np.linspace(0, 64, lr_data.shape[0])
    hr_grid = np.linspace(0, 64, hr_data.shape[0])
    sr_grid = np.linspace(0, 64, sr_data.shape[0])
    
    lr_X, lr_Y, lr_Z = np.meshgrid(lr_grid, lr_grid, lr_grid)
    hr_X, hr_Y, hr_Z = np.meshgrid(hr_grid, hr_grid, hr_grid)
    sr_X, sr_Y, sr_Z = np.meshgrid(sr_grid, sr_grid, sr_grid)
    
    # Get common color scale range for all plots
    vmin = min(lr_data.min(), hr_data.min(), sr_data.min())
    vmax = max(lr_data.max(), hr_data.max(), sr_data.max())
    
    # Low-resolution plot
    fig.add_trace(go.Volume(
        x=lr_X.flatten(),
        y=lr_Y.flatten(),
        z=lr_Z.flatten(),
        value=lr_data.flatten(),
        opacity=0.3,
        surface_count=15,
        colorscale='viridis',
        showscale=True,
    ), row=1, col=1)
    
    # High-resolution plot
    fig.add_trace(go.Volume(
        x=hr_X.flatten(),
        y=hr_Y.flatten(),
        z=hr_Z.flatten(),
        value=hr_data.flatten(),
        opacity=0.3,
        surface_count=15,
        colorscale='viridis',
        showscale=True,
    ), row=1, col=2)
    
    # Super-resolution plot
    fig.add_trace(go.Volume(
        x=sr_X.flatten(),
        y=sr_Y.flatten(),
        z=sr_Z.flatten(),
        value=sr_data.flatten(),
        opacity=0.3,
        surface_count=15,
        colorscale='viridis',
        showscale=True,
    ), row=1, col=3)

# Create figure
fig = make_subplots(
    rows=1, cols=3,
    specs=[[{'type': 'volume'}, {'type': 'volume'}, {'type': 'volume'}]],
    subplot_titles=("Low Resolution", "High Resolution", "Super Resolution")
)

# Get test samples
test_lr_files = sorted(glob.glob('test/lr_scalar-*.npy'))
sample_numbers = [int(f.split('-')[-1].split('.')[0]) for f in test_lr_files]

if sample_numbers:
    sample_str = f"{sample_numbers[0]:03d}"
    sample_str = "002"
    
    # Define paths
    lr_path = f'test/lr_scalar-{sample_str}.npy'
    hr_path = f'test/hr_scalar-{sample_str}.npy'
    sr_path = f'hr_scalar-{sample_str}_out.npy'
    
    if all(os.path.exists(p) for p in [lr_path, hr_path, sr_path]):
        load_and_plot_triplet(lr_path, hr_path, sr_path, fig)
        
        # Layout adjustments
        fig.update_layout(
            title_text=f"3D Volume Comparison (Test Sample #{sample_str})<br>LR vs HR vs SR",
            height=800,
            width=1800,
        )
        
        # Set all plots to range 0-64 and cube aspect ratio
        for i in range(1, 4):
            scene_key = f'scene{i if i > 1 else ""}'
            fig.update_layout({
                scene_key: dict(
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
        print("Missing one or more files:")
        print(f"LR: {lr_path} - {'Exists' if os.path.exists(lr_path) else 'Missing'}")
        print(f"HR: {hr_path} - {'Exists' if os.path.exists(hr_path) else 'Missing'}")
        print(f"SR: {sr_path} - {'Exists' if os.path.exists(sr_path) else 'Missing'}")
else:
    print("No test samples found!") 