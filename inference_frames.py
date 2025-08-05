import cv2
import numpy as np
import torch
import torchvision.transforms as T
import os
import glob
from pathlib import Path
from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image
from models.SpaTrackV2.models.predictor import Predictor
from models.SpaTrackV2.utils.visualizer import Visualizer
from models.SpaTrackV2.models.utils import get_points_on_a_grid
import argparse
from rich import print
from rich.console import Console

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track_mode", type=str, default="offline", choices=["offline", "online"])
    parser.add_argument("--data_type", type=str, default="RGB")
    parser.add_argument("--images_dir", type=str, required=True, help="Directory containing RGB images")
    parser.add_argument("--images_pattern", type=str, default="*.jpg", help="Pattern to match image files (e.g., '*.jpg', '*.png')")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory for results")
    parser.add_argument("--mask_path", type=str, default=None, help="Path to mask image (optional)")
    parser.add_argument("--grid_size", type=int, default=10, help="Grid size for initial points")
    parser.add_argument("--vo_points", type=int, default=756, help="Number of visual odometry points")
    parser.add_argument("--max_frames", type=int, default=100, help="Maximum number of frames to process")
    parser.add_argument("--fps", type=int, default=1, help="Frame sampling rate")
    parser.add_argument("--image_extensions", nargs="+", default=[".jpg", ".jpeg", ".png", ".bmp"], 
                       help="Valid image extensions")
    return parser.parse_args()

def natural_sort_key(filename):
    """Sort filenames naturally (e.g., img1.jpg, img2.jpg, img10.jpg)"""
    import re
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', filename)]

def load_images_from_directory(images_dir, pattern="*.jpg", max_frames=100, fps=1, extensions=None):
    """
    Load RGB images from a directory and convert to tensor format.
    
    Args:
        images_dir (str): Directory containing images
        pattern (str): Pattern to match image files
        max_frames (int): Maximum number of frames to load
        fps (int): Frame sampling rate (take every fps-th frame)
        extensions (list): Valid image extensions
    
    Returns:
        torch.Tensor: Video tensor of shape (T, C, H, W)
    """
    console = Console()
    
    if not os.path.exists(images_dir):
        raise ValueError(f"Images directory does not exist: {images_dir}")
    
    # Get all image files matching the pattern
    image_paths = glob.glob(os.path.join(images_dir, pattern))
    
    # If no files found with pattern, try all valid extensions
    if not image_paths and extensions:
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(images_dir, f"*{ext}")))
            image_paths.extend(glob.glob(os.path.join(images_dir, f"*{ext.upper()}")))
    
    if not image_paths:
        raise ValueError(f"No images found in {images_dir} with pattern {pattern}")
    
    # Sort naturally (handle numbered sequences correctly)
    image_paths.sort(key=lambda x: natural_sort_key(os.path.basename(x)))
    
    console.print(f"[bold cyan]Found {len(image_paths)} images in:[/] {images_dir}")
    console.print(f"[green]First few images:[/] {[os.path.basename(p) for p in image_paths[:5]]}")
    
    frames = []
    for i, image_path in enumerate(image_paths):
        if len(frames) >= max_frames:
            break
            
        # Sample frames according to fps
        if i % fps != 0:
            continue
            
        if i % 50 == 0:
            console.print(f"[green]Loading image {i}: {os.path.basename(image_path)}[/]")
        
        try:
            # Load image
            frame = cv2.imread(image_path)
            if frame is None:
                console.print(f"[yellow]Warning: Could not load {image_path}[/]")
                continue
                
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            
        except Exception as e:
            console.print(f"[red]Error loading {image_path}: {e}[/]")
            continue
    
    if not frames:
        raise ValueError("No valid images could be loaded")
    
    console.print(f"[bold magenta]Total frames loaded:[/] {len(frames)}")
    
    # Convert to tensor
    video_tensor = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float()
    console.print(f"[bold blue]Video tensor shape:[/] {video_tensor.shape}")
    
    return video_tensor

def load_images_from_list(image_paths, max_frames=100, fps=1):
    """
    Load RGB images from a list of paths and convert to tensor format.
    
    Args:
        image_paths (list): List of image file paths
        max_frames (int): Maximum number of frames to load
        fps (int): Frame sampling rate
    
    Returns:
        torch.Tensor: Video tensor of shape (T, C, H, W)
    """
    console = Console()
    
    if not image_paths:
        raise ValueError("No image paths provided")
    
    # Sort naturally if needed
    image_paths.sort(key=lambda x: natural_sort_key(os.path.basename(x)))
    
    console.print(f"[bold cyan]Processing {len(image_paths)} images[/]")
    
    frames = []
    for i, image_path in enumerate(image_paths):
        if len(frames) >= max_frames:
            break
            
        # Sample frames according to fps
        if i % fps != 0:
            continue
            
        if i % 50 == 0:
            console.print(f"[green]Loading image {i}: {os.path.basename(image_path)}[/]")
        
        try:
            # Load image
            frame = cv2.imread(image_path)
            if frame is None:
                console.print(f"[yellow]Warning: Could not load {image_path}[/]")
                continue
                
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            
        except Exception as e:
            console.print(f"[red]Error loading {image_path}: {e}[/]")
            continue
    
    if not frames:
        raise ValueError("No valid images could be loaded")
    
    console.print(f"[bold magenta]Total frames loaded:[/] {len(frames)}")
    
    # Convert to tensor
    video_tensor = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float()
    console.print(f"[bold blue]Video tensor shape:[/] {video_tensor.shape}")
    
    return video_tensor

def run_spatial_tracker_inference(image_paths_or_dir, **kwargs):
    """
    Run spatial tracker inference on a list of images or directory.
    
    Args:
        image_paths_or_dir: Either a list of image paths or a directory path
        **kwargs: Additional arguments (track_mode, mask_path, grid_size, etc.)
    
    Returns:
        dict: Results containing tracking data
    """
    console = Console()
    
    # Parse arguments
    track_mode = kwargs.get('track_mode', 'offline')
    mask_path = kwargs.get('mask_path', None)
    grid_size = kwargs.get('grid_size', 10)
    vo_points = kwargs.get('vo_points', 756)
    max_frames = kwargs.get('max_frames', 100)
    fps = kwargs.get('fps', 1)
    output_dir = kwargs.get('output_dir', 'results')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load images
    if isinstance(image_paths_or_dir, str):
        if os.path.isdir(image_paths_or_dir):
            # Load from directory
            pattern = kwargs.get('images_pattern', '*.jpg')
            extensions = kwargs.get('image_extensions', ['.jpg', '.jpeg', '.png', '.bmp'])
            video_tensor = load_images_from_directory(
                image_paths_or_dir, pattern, max_frames, fps, extensions
            )
        else:
            raise ValueError(f"Path is not a directory: {image_paths_or_dir}")
    elif isinstance(image_paths_or_dir, list):
        # Load from list of paths
        video_tensor = load_images_from_list(image_paths_or_dir, max_frames, fps)
    else:
        raise ValueError("Input must be either a directory path or list of image paths")
    
    console.print(f"[bold green]Loaded video tensor with shape:[/] {video_tensor.shape}")
    
    # Load VGGT4Track model for depth and pose estimation
    console.print("[bold cyan]Loading VGGT4Track model...[/]")
    vggt4track_model = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front")
    vggt4track_model.eval()
    vggt4track_model = vggt4track_model.to("cuda")
    
    # Preprocess and run depth/pose estimation
    console.print("[bold cyan]Running depth and pose estimation...[/]")
    processed_video = preprocess_image(video_tensor)[None]
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            predictions = vggt4track_model(processed_video.cuda()/255)
            extrinsic, intrinsic = predictions["poses_pred"], predictions["intrs"]
            depth_map, depth_conf = predictions["points_map"][..., 2], predictions["unc_metric"]
    
    # Extract results
    depth_tensor = depth_map.squeeze().cpu().numpy()
    extrs = extrinsic.squeeze().cpu().numpy()
    intrs = intrinsic.squeeze().cpu().numpy()
    video_tensor = processed_video.squeeze()
    unc_metric = depth_conf.squeeze().cpu().numpy() > 0.5
    
    # Load mask if provided
    if mask_path and os.path.exists(mask_path):
        console.print(f"[bold cyan]Loading mask from:[/] {mask_path}")
        mask = cv2.imread(mask_path)
        mask = cv2.resize(mask, (video_tensor.shape[2], video_tensor.shape[1]))
        mask = mask.sum(axis=-1) > 0
    else:
        mask = np.ones_like(video_tensor[0,0].numpy()) > 0
    
    # Load spatial tracker model
    console.print(f"[bold cyan]Loading spatial tracker model ({track_mode})...[/]")
    if track_mode == "offline":
        model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Offline")
    else:
        model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Online")
    
    model.spatrack.track_num = vo_points
    model.eval()
    model.to("cuda")
    
    # Initialize visualizer
    viser = Visualizer(save_dir=output_dir, grayscale=True, fps=10, pad_value=0, tracks_leave_trace=5)
    
    # Generate grid points
    frame_H, frame_W = video_tensor.shape[2:]
    grid_pts = get_points_on_a_grid(grid_size, (frame_H, frame_W), device="cpu")
    
    # Apply mask to grid points if available
    if mask_path and os.path.exists(mask_path):
        grid_pts_int = grid_pts[0].long()
        mask_values = mask[grid_pts_int[...,1], grid_pts_int[...,0]]
        grid_pts = grid_pts[:, mask_values]
    
    query_xyt = torch.cat([torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2)[0].numpy()
    
    # Run spatial tracking
    console.print("[bold cyan]Running spatial tracker inference...[/]")
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        (
            c2w_traj, intrs, point_map, conf_depth,
            track3d_pred, track2d_pred, vis_pred, conf_pred, video
        ) = model.forward(video_tensor, depth=depth_tensor,
                            intrs=intrs, extrs=extrs,
                            queries=query_xyt,
                            fps=1, full_point=False, iters_track=4,
                            query_no_BA=True, fixed_cam=False, stage=1, unc_metric=unc_metric,
                            support_frame=len(video_tensor)-1, replace_ratio=0.2)
    
    # Resize if needed
    max_size = 336
    h, w = video.shape[2:]
    scale = min(max_size / h, max_size / w)
    if scale < 1:
        console.print(f"[yellow]Resizing video from {(h, w)} to {(int(h * scale), int(w * scale))}[/]")
        new_h, new_w = int(h * scale), int(w * scale)
        video = T.Resize((new_h, new_w))(video)
        video_tensor = T.Resize((new_h, new_w))(video_tensor)
        point_map = T.Resize((new_h, new_w))(point_map)
        conf_depth = T.Resize((new_h, new_w))(conf_depth)
        track2d_pred[...,:2] = track2d_pred[...,:2] * scale
        intrs[:,:2,:] = intrs[:,:2,:] * scale
        if depth_tensor is not None:
            if isinstance(depth_tensor, torch.Tensor):
                depth_tensor = T.Resize((new_h, new_w))(depth_tensor)
            else:
                depth_tensor = T.Resize((new_h, new_w))(torch.from_numpy(depth_tensor))
    
    # Visualize results
    console.print("[bold cyan]Generating visualization...[/]")
    viser.visualize(video=video[None],
                    tracks=track2d_pred[None][...,:2],
                    visibility=vis_pred[None],
                    filename="inference_frames_result")
    
    # Prepare results for saving
    data_npz_load = {}
    data_npz_load["coords"] = (torch.einsum("tij,tnj->tni", c2w_traj[:,:3,:3], track3d_pred[:,:,:3].cpu()) + c2w_traj[:,:3,3][:,None,:]).numpy()
    data_npz_load["extrinsics"] = torch.inverse(c2w_traj).cpu().numpy()
    data_npz_load["intrinsics"] = intrs.cpu().numpy()
    depth_save = point_map[:,2,...]
    depth_save[conf_depth<0.5] = 0
    data_npz_load["depths"] = depth_save.cpu().numpy()
    data_npz_load["video"] = (video_tensor).cpu().numpy()/255
    data_npz_load["visibs"] = vis_pred.cpu().numpy()
    data_npz_load["unc_metric"] = conf_depth.cpu().numpy()
    
    # Save results
    result_path = os.path.join(output_dir, 'result.npz')
    np.savez(result_path, **data_npz_load)
    
    console.print(f"[bold green]Results saved to {output_dir}[/]")
    console.print(f"[bold yellow]To visualize results with tapip3d, run:[/] python tapip3d_viz.py {result_path}")
    
    return data_npz_load

if __name__ == "__main__":
    args = parse_args()
    
    # Run inference
    results = run_spatial_tracker_inference(
        args.images_dir,
        track_mode=args.track_mode,
        mask_path=args.mask_path,
        grid_size=args.grid_size,
        vo_points=args.vo_points,
        max_frames=args.max_frames,
        fps=args.fps,
        output_dir=args.output_dir,
        images_pattern=args.images_pattern,
        image_extensions=args.image_extensions
    )
    
    print(f"[bold green]Inference completed successfully![/]") 