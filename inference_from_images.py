import os
import torch
import numpy as np
import argparse
import random

# add to path
import sys
# add one dir out
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from consecutive_images_dataset import ConsecutiveImagesDataset, get_default_transforms
from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image
from models.SpaTrackV2.models.predictor import Predictor
from models.SpaTrackV2.utils.visualizer import Visualizer
from models.SpaTrackV2.models.utils import get_points_on_a_grid
import torchvision.transforms as T


def run_inference_on_batches(
    image_dir,
    batch_size,
    out_dir,
    track_mode="offline",
    grid_size=10,
    vo_points=756,
    device="cuda",
    max_batches=3
):
    os.makedirs(out_dir, exist_ok=True)
    dataset = ConsecutiveImagesDataset(
        image_dir=image_dir,
        batch_size=batch_size,
        transform=get_default_transforms(),
        start_frame_idx=300
    )
    print(f"Loaded dataset with {len(dataset)} batches of {batch_size} images.")

    vggt4track_model = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front")
    vggt4track_model.eval()
    vggt4track_model = vggt4track_model.to(device)

    if track_mode == "offline":
        model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Offline")
    else:
        model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Online")
    model.spatrack.track_num = vo_points
    model.eval()
    model.to(device)

    viser = Visualizer(save_dir=out_dir, grayscale=True, fps=10, pad_value=0, tracks_leave_trace=5)

    total_batches = len(dataset)
    if max_batches > total_batches:
        max_batches = total_batches
    random_indices = random.sample(range(total_batches), max_batches)
    print(f"Randomly selected batch indices: {random_indices}")
    save_idx = 0

    for batch_count, batch_idx in enumerate(random_indices):
        print(f"\nProcessing random batch {batch_count+1}/{max_batches} (dataset index {batch_idx})...")
        batch_tensor = dataset[batch_idx]
          # Debugging breakpoint

        # Visualize video_tensor before sending to model
        # import matplotlib.pyplot as plt
        # import pdb; pdb.set_trace()  # Debugging breakpoint
        video_tensor = batch_tensor  # (T, C, H, W)
        batch_size = video_tensor.shape[0]
        ncols = min(5, batch_size)
        nrows = (batch_size + ncols - 1) // ncols
        import time
        video_tensor = preprocess_image(video_tensor)[None]  # (1, T, C, H, W)
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                predictions = vggt4track_model(video_tensor.to(device))
                extrinsic, intrinsic = predictions["poses_pred"], predictions["intrs"]
                depth_map, depth_conf = predictions["points_map"][..., 2], predictions["unc_metric"]
                # Save all points_map, not just depth
                points_map = predictions["points_map"]

        # Visualize depth_map as an animation
        # show_depth_map_animation(depth_map)

        depth_tensor = depth_map.squeeze().cpu().numpy()
        extrs = extrinsic.squeeze().cpu().numpy()
        # extr_file is inverse of extrs, using numpy of extrs
        extrs_inv = np.linalg.inv(extrs)
        intrs = intrinsic.squeeze().cpu().numpy()
        video_tensor = video_tensor.squeeze()
        unc_metric = depth_conf.squeeze().cpu().numpy() > 0.5
        conf = depth_conf.squeeze().cpu().numpy()
        depth_tensor_write = depth_tensor.copy()
        depth_tensor_write[conf<0.5] = 0
        pre_refinement_data_npz = {}

        # Use data from predictions (lines 90) for saving
        pre_refinement_data_npz["extrinsics"] = extrs_inv
        pre_refinement_data_npz["intrinsics"] = intrs
        pre_refinement_data_npz["depths"] = depth_tensor_write
        pre_refinement_data_npz["video"] = video_tensor.cpu().numpy() if hasattr(video_tensor, 'cpu') else np.array(video_tensor)
        pre_refinement_data_npz["unc_metric"] = depth_conf.squeeze().cpu().numpy()
        pre_refinement_data_npz["points_map"] = points_map.squeeze().cpu().numpy()
        
        # Optionally add more from predictions if needed
        # save data_npz_load
        for key, value in pre_refinement_data_npz.items():
            print(f"{key}: {value.shape if isinstance(value, np.ndarray) else type(value)}")

        np.savez(os.path.join(out_dir, f'result_batch_pre_refinement{save_idx}.npz'), **pre_refinement_data_npz)

        # No mask support for images, use all ones
        mask = np.ones_like(video_tensor[0,0].numpy()) > 0

        frame_H, frame_W = video_tensor.shape[2:]
        grid_pts = get_points_on_a_grid(grid_size, (frame_H, frame_W), device="cpu")
        query_xyt = torch.cat([torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2)[0].numpy()

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
            max_size = 336
            h, w = video.shape[2:]
            scale = min(max_size / h, max_size / w)
            if scale < 1:
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
            viser.visualize(video=video[None],
                            tracks=track2d_pred[None][...,:2],
                            visibility=vis_pred[None],filename=f"batch_{batch_idx}")
            
            import pdb; pdb.set_trace()
            data_npz_load = {}
            data_npz_load["coords"] = (torch.einsum("tij,tnj->tni", c2w_traj[:,:3,:3], track3d_pred[:,:,:3].cpu()) + c2w_traj[:,:3,3][:,None,:]).numpy()
            data_npz_load["extrinsics"] = torch.inverse(c2w_traj).cpu().numpy()
            data_npz_load["intrinsics"] = intrs.cpu().numpy()
            depth_save = point_map[:,2,...]
            depth_save[conf_depth<0.5] = 0
            data_npz_load["depths"] = depth_save.cpu().numpy()
            data_npz_load["video"] = (video_tensor).cpu().numpy()
            data_npz_load["visibs"] = vis_pred.cpu().numpy()
            data_npz_load["unc_metric"] = conf_depth.cpu().numpy()

            # print all the shapes of data_npz_load
            for key, value in data_npz_load.items():
                print(f"{key}: {value.shape if isinstance(value, np.ndarray) else type(value)}")
            # Save results

            np.savez(os.path.join(out_dir, f'result_batch_{save_idx}.npz'), **data_npz_load)
            print(f"Results saved to {out_dir}.\nTo visualize them with tapip3d, run: python tapip3d_viz.py {os.path.join(out_dir, f'result_batch_{save_idx}.npz')}")
            save_idx += 1
            # input("Press Enter to continue to the next batch...")
        if batch_count+1 >= max_batches:
            print("Reached max_batches limit, stopping early.")
            break

def show_depth_map_animation(depth_map):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    dm = depth_map.detach().cpu().numpy()
    if dm.ndim == 4:
        dm = dm.squeeze(1)  # (T, H, W)
    nframes = dm.shape[0]
    vmin = np.percentile(dm, 2)
    vmax = np.percentile(dm, 98)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(dm[0], cmap='plasma', vmin=vmin, vmax=vmax)
    ax.set_title('Depth Animation')
    ax.axis('off')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def update(frame):
        im.set_data(dm[frame])
        ax.set_title(f'Depth Frame {frame}')
        return [im]

    ani = FuncAnimation(fig, update, frames=nframes, interval=200, blit=True, repeat=True)
    plt.show()
    input("Above: depth map animation. Press Enter to continue...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SpatialTrackerV2 inference on consecutive image batches.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--batch_size", type=int, default=20, help="Number of consecutive images per batch")
    parser.add_argument("--out_dir", type=str, default="results_from_images", help="Output directory")
    parser.add_argument("--track_mode", type=str, default="offline", choices=["offline", "online"], help="Tracking mode")
    parser.add_argument("--grid_size", type=int, default=10, help="Grid size for query points")
    parser.add_argument("--vo_points", type=int, default=756, help="Number of points for tracking")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--max_batches", type=int, default=10, help="Max number of batches to process")
    args = parser.parse_args()

    run_inference_on_batches(
        image_dir=args.image_dir,
        batch_size=args.batch_size,
        out_dir=args.out_dir,
        track_mode=args.track_mode,
        grid_size=args.grid_size,
        vo_points=args.vo_points,
        device=args.device,
        max_batches=args.max_batches
    )
