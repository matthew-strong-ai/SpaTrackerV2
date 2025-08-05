import os
import torch
from torch.utils.data import Dataset, DataLoader

import sys
# add one dir out
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from consecutive_images_dataset import ConsecutiveImagesDataset, get_default_transforms
from typing import List, Optional, Callable, Tuple
import matplotlib.pyplot as plt
import numpy as np
import random

class SequenceLearningDataset(Dataset):
    """
    Dataset that combines ConsecutiveImagesDataset from multiple folders for sequence learning.
    Each item returns (X, y) where:
    - X: first m frames from a sequence 
    - y: last n frames from the same sequence
    
    Args:
        image_dirs: List of directories containing image sequences
        m: Number of input frames (first m frames)
        n: Number of target frames (last n frames)  
        transform: Optional transform to apply to images
        start_frame_idx: Starting frame index for each sequence
    """
    def __init__(
        self,
        image_dirs: List[str],
        m: int,  # Input frames
        n: int,  # Target frames
        transform: Optional[Callable] = None,
        start_frame_idx: int = 0
    ):
        self.m = m
        self.n = n
        self.total_frames = m + n  # Total frames needed per sequence
        self.datasets = []
        self.indices = []  # (dataset_idx, batch_idx)

        for d_idx, image_dir in enumerate(image_dirs):
            ds = ConsecutiveImagesDataset(
                image_dir=image_dir,
                batch_size=self.total_frames,  # Need m+n consecutive frames
                transform=transform or get_default_transforms(),
                start_frame_idx=start_frame_idx
            )
            self.datasets.append(ds)
            for i in range(len(ds)):
                self.indices.append((d_idx, i))

            print(f"Loaded {len(ds)} sequences from {image_dir} (m={m}, n={n})")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        d_idx, batch_idx = self.indices[idx]
        full_sequence = self.datasets[d_idx][batch_idx]  # Shape: (m+n, C, H, W)
        
        # Split into input (first m frames) and target (last n frames)
        X = full_sequence[:self.m]  # Shape: (m, C, H, W)
        y = full_sequence[-self.n:] # Shape: (n, C, H, W)
        
        return X, y

def visualize_random_samples(dataset, n=5):
    """
    Visualize n random samples from the sequence learning dataset.
    Each sample contains (X, y) where X are input frames and y are target frames.
    """
    if len(dataset) == 0:
        print("Dataset is empty.")
        return
    indices = random.sample(range(len(dataset)), min(n, len(dataset)))
    for idx in indices:
        X, y = dataset[idx]  # X: (m, C, H, W), y: (n, C, H, W)
        
        # Convert to numpy if needed
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        
        m, n = X.shape[0], y.shape[0]
        total_frames = m + n
        
        # Create visualization layout
        ncols = min(8, total_frames)
        nrows = (total_frames + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.5))
        axes = np.array(axes).reshape(nrows, ncols)
        
        # Plot input frames (X)
        for i in range(m):
            row, col = divmod(i, ncols)
            ax = axes[row, col]
            img = X[i].transpose(1, 2, 0)
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f'Input {i}', color='blue')
            ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, 
                                     fill=False, edgecolor='blue', linewidth=2))
        
        # Plot target frames (y)  
        for i in range(n):
            frame_idx = m + i
            row, col = divmod(frame_idx, ncols)
            ax = axes[row, col]
            img = y[i].transpose(1, 2, 0)
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f'Target {i}', color='red')
            ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, 
                                     fill=False, edgecolor='red', linewidth=2))
        
        # Hide unused axes
        for i in range(total_frames, nrows * ncols):
            row, col = divmod(i, ncols)
            axes[row, col].axis('off')
        
        plt.suptitle(f"Sequence {idx}: {m} Input Frames (blue) → {n} Target Frames (red)")
        plt.tight_layout()
        plt.show()
        input("Above: sequence visualization. Press Enter to continue...")

# Example training loop
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train with SequenceLearningDataset (root dir with subfolders)")
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory containing subfolders of images')
    parser.add_argument('--m', type=int, default=8, help='Number of input frames')
    parser.add_argument('--n', type=int, default=5, help='Number of target frames') 
    parser.add_argument('--epochs', type=int, default=2)
    args = parser.parse_args()


    # Find all subdirectories in root_dir
    image_dirs = [os.path.join(args.root_dir, d) for d in os.listdir(args.root_dir)
                  if os.path.isdir(os.path.join(args.root_dir, d))]
    print(f"Found {len(image_dirs)} subfolders:")
    for d in image_dirs:
        print(f"  {d}")

    dataset = SequenceLearningDataset(
        image_dirs=image_dirs,
        m=args.m,
        n=args.n,
        transform=get_default_transforms()
    )

    # print length of dataset
    print(f"Total sequences in dataset: {len(dataset)}")



    # Visualization: show n random sequences
    visualize_random_samples(dataset, n=10)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    # Dummy sequence-to-sequence model for demonstration
    input_dim = 3 * 540 * 960 * args.m  # m input frames
    output_dim = 3 * 540 * 960 * args.n  # n target frames
    
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(input_dim, 512),
        torch.nn.ReLU(), 
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, output_dim)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        for i, (X, y) in enumerate(dataloader):
            # X: (1, m, C, H, W), y: (1, n, C, H, W)
            X = X.squeeze(0)  # (m, C, H, W)
            y = y.squeeze(0)  # (n, C, H, W)
            
            # Flatten for simple demo model
            X_flat = X.flatten()
            y_flat = y.flatten()

            output = model(X_flat.unsqueeze(0))  # Add batch dim
            loss = criterion(output, y_flat.unsqueeze(0))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % 10 == 0:
                print(f"  Sequence {i}: Loss = {loss.item():.4f}")
    print("Training complete.")
