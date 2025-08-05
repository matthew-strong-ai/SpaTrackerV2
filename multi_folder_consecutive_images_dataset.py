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

class MultiFolderConsecutiveImagesDataset(Dataset):
    """
    Dataset that combines ConsecutiveImagesDataset from multiple folders.
    Each item is a batch of consecutive images from one of the folders.
    """
    def __init__(
        self,
        image_dirs: List[str],
        batch_size: int,
        transform: Optional[Callable] = None,
        start_frame_idx: int = 0
    ):
        self.datasets = []
        self.indices = []  # (dataset_idx, batch_idx)

        for d_idx, image_dir in enumerate(image_dirs):
            ds = ConsecutiveImagesDataset(
                image_dir=image_dir,
                batch_size=batch_size,
                transform=transform or get_default_transforms(),
                start_frame_idx=start_frame_idx
            )
            self.datasets.append(ds)
            for i in range(len(ds)):
                self.indices.append((d_idx, i))

            print(f"Loaded {len(ds)} batches from {image_dir}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        d_idx, batch_idx = self.indices[idx]
        return self.datasets[d_idx][batch_idx]

def visualize_random_samples(dataset, n=5):
    """
    Visualize n random samples from the dataset using matplotlib.
    Each sample is a batch of consecutive images (T, C, H, W).
    """
    if len(dataset) == 0:
        print("Dataset is empty.")
        return
    indices = random.sample(range(len(dataset)), min(n, len(dataset)))
    for idx in indices:
        batch = dataset[idx]  # (T, C, H, W)
        if isinstance(batch, torch.Tensor):
            batch = batch.cpu().numpy()
        T = batch.shape[0]
        ncols = min(5, T)
        nrows = (T + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
        axes = np.array(axes).reshape(nrows, ncols)
        for i in range(T):
            row, col = divmod(i, ncols)
            ax = axes[row, col]
            img = batch[i].transpose(1, 2, 0)
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f'Frame {i}')
        for i in range(T, nrows * ncols):
            row, col = divmod(i, ncols)
            axes[row, col].axis('off')
        plt.suptitle(f"Sample batch idx {idx}")
        plt.tight_layout()
        plt.show()
        input("Above: batch visualization. Press Enter to continue...")

# Example training loop
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train with MultiFolderConsecutiveImagesDataset (root dir with subfolders)")
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory containing subfolders of images')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=2)
    args = parser.parse_args()


    # Find all subdirectories in root_dir
    image_dirs = [os.path.join(args.root_dir, d) for d in os.listdir(args.root_dir)
                  if os.path.isdir(os.path.join(args.root_dir, d))]
    print(f"Found {len(image_dirs)} subfolders:")
    for d in image_dirs:
        print(f"  {d}")

    dataset = MultiFolderConsecutiveImagesDataset(
        image_dirs=image_dirs,
        batch_size=args.batch_size,
        transform=get_default_transforms()
    )

    # print length of dataset
    print(f"Total batches in dataset: {len(dataset)}")

    # Visualization: show n random batches
    visualize_random_samples(dataset, n=5)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    # Dummy model for demonstration
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(3 * 540 * 960 * args.batch_size, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        for i, batch in enumerate(dataloader):
            # batch: (1, batch_size, C, H, W)
            # batch = batch.squeeze(0)  # (batch_size, C, H, W)
            # Dummy target
            target = torch.randn(1)

            output = model(batch)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(f"  Batch {i}: Loss = {loss.item():.4f}")
    print("Training complete.")
