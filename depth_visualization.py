"""
Depth Visualization Tools
========================

Clean utilities for creating depth videos and 3D point cloud visualizations
from SpaTrackerV2 inference results.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import os
from typing import Optional, Tuple, Union
import torch


def create_depth_video(
    depth_maps: np.ndarray,
    output_path: str,
    fps: int = 10,
    colormap: str = 'plasma',
    normalize_per_frame: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    add_colorbar: bool = True,
    resolution: Optional[Tuple[int, int]] = None
) -> str:
    """
    Create a video from a sequence of depth maps.
    
    Args:
        depth_maps: Array of shape (T, H, W) containing depth values
        output_path: Path to save the output video
        fps: Frames per second for the video
        colormap: Matplotlib colormap name
        normalize_per_frame: If True, normalize each frame independently
        vmin, vmax: Depth value range for normalization (if None, auto-computed)
        add_colorbar: Whether to add a colorbar to each frame
        resolution: Output resolution (width, height). If None, uses original size
        
    Returns:
        Path to the created video file
    """
    
    if depth_maps.ndim != 3:
        raise ValueError(f"Expected depth_maps shape (T, H, W), got {depth_maps.shape}")
    
    T, H, W = depth_maps.shape
    
    # Set up output path
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Set normalization range
    if not normalize_per_frame:
        if vmin is None:
            vmin = np.percentile(depth_maps, 2)
        if vmax is None:
            vmax = np.percentile(depth_maps, 98)
    
    # Set up video writer
    if resolution is None:
        if add_colorbar:
            # Add space for colorbar
            video_width = W + 100  # Extra space for colorbar
            video_height = H
        else:
            video_width = W
            video_height = H
    else:
        video_width, video_height = resolution
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (video_width, video_height))
    
    print(f"Creating depth video: {T} frames at {fps}fps...")
    
    # Create each frame
    for t in range(T):
        frame_depth = depth_maps[t]
        
        # Normalize current frame
        if normalize_per_frame:
            frame_vmin = np.percentile(frame_depth, 2)
            frame_vmax = np.percentile(frame_depth, 98)
        else:
            frame_vmin, frame_vmax = vmin, vmax
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
        
        # Plot depth map
        im = ax.imshow(frame_depth, cmap=colormap, vmin=frame_vmin, vmax=frame_vmax)
        ax.set_title(f'Depth Frame {t}/{T-1}', fontsize=14)
        ax.axis('off')
        
        # Add colorbar
        if add_colorbar:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Depth', rotation=270, labelpad=20)
        
        plt.tight_layout()
        
        # Convert matplotlib figure to numpy array
        fig.canvas.draw()
        
        # Handle different matplotlib versions
        try:
            # Newer matplotlib versions
            buffer = fig.canvas.buffer_rgba()
            frame_array = np.asarray(buffer)
            # Convert RGBA to RGB
            frame_array = frame_array[:, :, :3]
        except AttributeError:
            try:
                # Older matplotlib versions
                frame_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                frame_array = frame_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            except AttributeError:
                # Alternative for some matplotlib versions
                frame_array = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
                frame_array = frame_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
                # Convert ARGB to RGB
                frame_array = frame_array[:, :, 1:4]
        
        # Resize if needed
        if resolution is not None:
            frame_array = cv2.resize(frame_array, resolution)
        
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
        
        # Write frame to video
        video_writer.write(frame_bgr)
        
        plt.close(fig)
        
        if (t + 1) % 10 == 0:
            print(f"  Processed {t + 1}/{T} frames")
    
    video_writer.release()
    print(f"Depth video saved to: {output_path}")
    
    return str(output_path)


def depth_to_point_cloud(
    depth_map: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: Optional[np.ndarray] = None,
    rgb_image: Optional[np.ndarray] = None,
    max_depth: float = 10.0,
    min_depth: float = 0.1
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Convert depth map to 3D point cloud.
    
    Args:
        depth_map: Depth values (H, W)
        intrinsics: Camera intrinsic matrix (3, 3) or (3,) with [fx, fy, cx, cy] format
        extrinsics: Camera extrinsic matrix (4, 4) for world coordinates
        rgb_image: RGB image (H, W, 3) for coloring points
        max_depth: Maximum depth to include
        min_depth: Minimum depth to include
        
    Returns:
        points: 3D points array (N, 3)
        colors: RGB colors array (N, 3) if rgb_image provided, else None
    """
    
    H, W = depth_map.shape
    
    # Handle different intrinsic formats and dimensionalities
    # First, handle batch and temporal dimensions if present
    if intrinsics.ndim > 2:
        # Handle cases like (1, T, 3, 3) or (B, T, 3, 3) or (T, 3, 3)
        while intrinsics.ndim > 2:
            intrinsics = intrinsics[0]  # Take first batch/frame
    
    if intrinsics.shape == (3,):
        # Assume [fx, fy, cx] - use center for cy
        fx, fy = intrinsics[0], intrinsics[1]
        cx = intrinsics[2] if len(intrinsics) > 2 else W // 2
        cy = H // 2
    elif intrinsics.shape == (4,):
        fx, fy, cx, cy = intrinsics
    elif intrinsics.shape == (3, 3):
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    else:
        raise ValueError(f"Unsupported intrinsics shape after processing: {intrinsics.shape}. Expected (3,), (4,), or (3,3)")
    
    # Create coordinate grids
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    
    # Filter by depth range
    valid_mask = (depth_map > min_depth) & (depth_map < max_depth) & (depth_map > 0)
    
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    depth_valid = depth_map[valid_mask]
    
    # Convert to 3D points
    X = (x_valid - cx) * depth_valid / fx
    Y = (y_valid - cy) * depth_valid / fy
    Z = depth_valid
    
    points = np.stack([X, Y, Z], axis=1)
    
    # Apply extrinsics if provided
    if extrinsics is not None:
        # Convert to homogeneous coordinates
        points_homo = np.hstack([points, np.ones((points.shape[0], 1))])
        # Transform to world coordinates
        points = (extrinsics @ points_homo.T)[:3].T
    
    # Extract colors if RGB image provided
    colors = None
    if rgb_image is not None:
        if rgb_image.shape[:2] != (H, W):
            raise ValueError(f"RGB image shape {rgb_image.shape[:2]} doesn't match depth shape {(H, W)}")
        colors = rgb_image[valid_mask]
        if colors.max() <= 1.0:
            colors = (colors * 255).astype(np.uint8)
    
    return points, colors


def create_point_cloud_visualization(
    depth_maps: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: Optional[np.ndarray] = None,
    rgb_images: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
    max_depth: float = 10.0,
    subsample_factor: int = 4,
    frame_indices: Optional[list] = None
) -> str:
    """
    Create 3D point cloud visualization from depth sequence.
    
    Args:
        depth_maps: Depth sequence (T, H, W)
        intrinsics: Camera intrinsics (T, 3, 3) or (3, 3) or (3,)
        extrinsics: Camera extrinsics (T, 4, 4) or (4, 4)
        rgb_images: RGB images (T, H, W, 3)
        output_path: Path to save visualization
        max_depth: Maximum depth for points
        subsample_factor: Factor to subsample points for performance
        frame_indices: Specific frames to visualize (if None, use all)
        
    Returns:
        Path to saved visualization or 'interactive' if shown interactively
    """
    
    T = depth_maps.shape[0]
    
    if frame_indices is None:
        frame_indices = list(range(T))
    
    all_points = []
    all_colors = []
    
    print(f"Converting {len(frame_indices)} depth frames to point clouds...")
    
    for i, frame_idx in enumerate(frame_indices):
        depth_frame = depth_maps[frame_idx]
        
        # Get intrinsics for this frame
        if intrinsics.ndim == 3:
            frame_intrinsics = intrinsics[frame_idx]
        else:
            frame_intrinsics = intrinsics
        
        # Get extrinsics for this frame
        frame_extrinsics = None
        if extrinsics is not None:
            if extrinsics.ndim == 3:
                frame_extrinsics = extrinsics[frame_idx]
            else:
                frame_extrinsics = extrinsics
        
        # Get RGB for this frame
        frame_rgb = None
        if rgb_images is not None:
            frame_rgb = rgb_images[frame_idx]
        
        # Convert to point cloud
        points, colors = depth_to_point_cloud(
            depth_frame, frame_intrinsics, frame_extrinsics, frame_rgb, max_depth
        )
        
        # Subsample for performance
        if subsample_factor > 1:
            indices = np.arange(0, len(points), subsample_factor)
            points = points[indices]
            if colors is not None:
                colors = colors[indices]
        
        # Add frame offset for temporal visualization
        if len(frame_indices) > 1:
            points[:, 0] += i * 2.0  # Offset in X direction
        
        all_points.append(points)
        if colors is not None:
            all_colors.append(colors)
        
        if (i + 1) % 5 == 0:
            print(f"  Processed {i + 1}/{len(frame_indices)} frames")
    
    # Combine all points
    combined_points = np.vstack(all_points)
    combined_colors = np.vstack(all_colors) if all_colors else None
    
    print(f"Total points: {len(combined_points):,}")
    
    # Create 3D visualization
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    if combined_colors is not None:
        colors_normalized = combined_colors / 255.0 if combined_colors.max() > 1 else combined_colors
        scatter = ax.scatter(
            combined_points[:, 0], combined_points[:, 1], combined_points[:, 2],
            c=colors_normalized, s=0.1, alpha=0.6
        )
    else:
        scatter = ax.scatter(
            combined_points[:, 0], combined_points[:, 1], combined_points[:, 2],
            c=combined_points[:, 2], cmap='plasma', s=0.1, alpha=0.6
        )
        plt.colorbar(scatter, ax=ax, shrink=0.5, label='Depth')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z (Depth)')
    ax.set_title(f'3D Point Cloud Visualization ({len(frame_indices)} frames)')
    
    # Set equal aspect ratio
    max_range = np.array([
        combined_points[:, 0].max() - combined_points[:, 0].min(),
        combined_points[:, 1].max() - combined_points[:, 1].min(),
        combined_points[:, 2].max() - combined_points[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (combined_points[:, 0].max() + combined_points[:, 0].min()) * 0.5
    mid_y = (combined_points[:, 1].max() + combined_points[:, 1].min()) * 0.5
    mid_z = (combined_points[:, 2].max() + combined_points[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Point cloud visualization saved to: {output_path}")
        plt.close()
        return str(output_path)
    else:
        plt.show()
        return 'interactive'


def create_depth_visualizations_from_inference(
    depth_maps: Union[np.ndarray, torch.Tensor],
    intrinsics: Union[np.ndarray, torch.Tensor],
    extrinsics: Optional[Union[np.ndarray, torch.Tensor]] = None,
    rgb_images: Optional[Union[np.ndarray, torch.Tensor]] = None,
    output_dir: str = "depth_visualizations",
    video_fps: int = 10,
    max_depth: float = 10.0,
    subsample_factor: int = 4
) -> dict:
    """
    Convenience function to create both depth video and 3D visualization from inference results.
    
    Args:
        depth_maps: Depth sequence from inference
        intrinsics: Camera intrinsics from inference
        extrinsics: Camera extrinsics from inference
        rgb_images: RGB images (optional)
        output_dir: Directory to save outputs
        video_fps: FPS for depth video
        max_depth: Maximum depth for point cloud
        subsample_factor: Subsampling for point cloud
        
    Returns:
        Dictionary with paths to created files
    """
    
    # Convert tensors to numpy if needed
    if torch.is_tensor(depth_maps):
        depth_maps = depth_maps.detach().cpu().numpy()
    if torch.is_tensor(intrinsics):
        intrinsics = intrinsics.detach().cpu().numpy()
    if extrinsics is not None and torch.is_tensor(extrinsics):
        extrinsics = extrinsics.detach().cpu().numpy()
    if rgb_images is not None and torch.is_tensor(rgb_images):
        rgb_images = rgb_images.detach().cpu().numpy()
    
    # Handle different depth map shapes
    if depth_maps.ndim == 4:
        depth_maps = depth_maps.squeeze(1)  # Remove channel dimension if present
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Create depth video
    print("Creating depth video...")
    video_path = create_depth_video(
        depth_maps,
        output_dir / "depth_sequence.mp4",
        fps=video_fps,
        colormap='plasma'
    )
    results['video_path'] = video_path
    
    # Create point cloud visualization
    print("Creating 3D point cloud visualization...")
    pointcloud_path = create_point_cloud_visualization(
        depth_maps,
        intrinsics,
        extrinsics,
        rgb_images,
        output_dir / "point_cloud_3d.png",
        max_depth=max_depth,
        subsample_factor=subsample_factor,
        frame_indices=list(range(0, len(depth_maps), max(1, len(depth_maps) // 10)))  # Sample 10 frames
    )
    results['pointcloud_path'] = pointcloud_path
    
    print(f"\nDepth visualizations created in: {output_dir}")
    print(f"  - Video: {results['video_path']}")
    print(f"  - Point cloud: {results['pointcloud_path']}")
    
    return results


# Integration function for inference_from_images.py
def add_depth_visualizations_to_inference(
    depth_map: torch.Tensor,
    intrinsic: torch.Tensor,
    extrinsic: Optional[torch.Tensor] = None,
    video_tensor: Optional[torch.Tensor] = None,
    output_dir: str = "depth_viz",
    batch_idx: int = 0
) -> dict:
    """
    Integration function to be called from inference_from_images.py around line 100.
    
    Args:
        depth_map: Depth maps from VGGT4Track model
        intrinsic: Intrinsics from VGGT4Track model  
        extrinsic: Extrinsics from VGGT4Track model
        video_tensor: Original video frames (optional, for RGB coloring)
        output_dir: Output directory
        batch_idx: Batch index for naming
        
    Returns:
        Dictionary with visualization paths
    """
    
    # Prepare RGB images if available
    rgb_images = None
    if video_tensor is not None:
        rgb_images = video_tensor.detach().cpu().numpy()
        if rgb_images.ndim == 4 and rgb_images.shape[1] == 3:
            # Convert from (T, C, H, W) to (T, H, W, C)
            rgb_images = rgb_images.transpose(0, 2, 3, 1)
        # Ensure values are in [0, 1] range
        if rgb_images.max() > 1.0:
            rgb_images = rgb_images / 255.0
    
    batch_output_dir = f"{output_dir}/batch_{batch_idx}"
    
    return create_depth_visualizations_from_inference(
        depth_map,
        intrinsic,
        extrinsic,
        rgb_images,
        batch_output_dir,
        video_fps=10,
        max_depth=10.0,
        subsample_factor=4
    )


if __name__ == "__main__":
    # Example usage
    print("Depth Visualization Tools")
    print("This module provides functions for creating depth videos and 3D point cloud visualizations.")
    print("Import and use the functions in your inference script.") 