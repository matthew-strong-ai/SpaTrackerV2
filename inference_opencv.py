import cv2
import numpy as np
import torch
import torchvision.transforms as T
import os
from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image
from models.SpaTrackV2.models.predictor import Predictor
from models.SpaTrackV2.utils.visualizer import Visualizer
from models.SpaTrackV2.models.utils import get_points_on_a_grid
import argparse
from rich import print

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track_mode", type=str, default="offline")
    parser.add_argument("--data_type", type=str, default="RGB")
    parser.add_argument("--data_dir", type=str, default="examples")
    parser.add_argument("--video_name", type=str, default="snowboard")
    parser.add_argument("--grid_size", type=int, default=10)
    parser.add_argument("--vo_points", type=int, default=756)
    parser.add_argument("--fps", type=int, default=1)
    return parser.parse_args()

def read_video_opencv(video_path, max_frames=100, fps=1):
    from rich.console import Console
    console = Console()
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    console.print(f"[bold cyan]Opening video:[/] {video_path}")
    while cap.isOpened() and len(frames) < (max_frames * 10):
        ret, frame = cap.read()
        if not ret:
            console.print(f"[yellow]End of video or read error at frame {count}.[/]")
            break
        if count % 50 == 0:
            console.print(f"[green]Reading frame {count}...[/]")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        count += 1
    cap.release()
    console.print(f"[bold magenta]Total frames read:[/] {len(frames)}")
    video_tensor = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float()
    video_tensor = video_tensor[::fps].float()[:max_frames]
    console.print(f"[bold blue]Video tensor shape after sampling:[/] {video_tensor.shape}")
    return video_tensor

if __name__ == "__main__":
    args = parse_args()
    out_dir = os.path.join(args.data_dir, "results")
    os.makedirs(out_dir, exist_ok=True)
    fps = int(args.fps)
    mask_dir = os.path.join(args.data_dir, f"{args.video_name}.png")
    video_path = os.path.join(args.data_dir, f"{args.video_name}.mp4")

    import pdb; pdb.set_trace()  # Debugging breakpoint

    vggt4track_model = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front")
    vggt4track_model.eval()
    vggt4track_model = vggt4track_model.to("cuda")

    print("Reading video with OpenCV...")
    video_tensor = read_video_opencv(video_path, max_frames=100, fps=fps)
    print(f"Video tensor shape: {video_tensor.shape}")

    print("Processing tensor")
    video_tensor = preprocess_image(video_tensor)[None]
    print(f"Video tensor shape after preprocessing: {video_tensor.shape}"   )
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            predictions = vggt4track_model(video_tensor.cuda()/255)
            extrinsic, intrinsic = predictions["poses_pred"], predictions["intrs"]
            depth_map, depth_conf = predictions["points_map"][..., 2], predictions["unc_metric"]

    depth_tensor = depth_map.squeeze().cpu().numpy()
    extrs = np.eye(4)[None].repeat(len(depth_tensor), axis=0)
    extrs = extrinsic.squeeze().cpu().numpy()
    intrs = intrinsic.squeeze().cpu().numpy()
    video_tensor = video_tensor.squeeze()
    unc_metric = depth_conf.squeeze().cpu().numpy() > 0.5
    data_npz_load = {}

    if os.path.exists(mask_dir):
        mask = cv2.imread(mask_dir)
        mask = cv2.resize(mask, (video_tensor.shape[2], video_tensor.shape[1]))
        mask = mask.sum(axis=-1) > 0
    else:
        mask = np.ones_like(video_tensor[0,0].numpy()) > 0

    viz = True
    # Model selection
    if args.track_mode == "offline":
        model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Offline")
    else:
        model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Online")
    model.spatrack.track_num = args.vo_points
    model.eval()
    model.to("cuda")

    viser = Visualizer(save_dir=out_dir, grayscale=True, fps=10, pad_value=0, tracks_leave_trace=5)
    grid_size = args.grid_size
    frame_H, frame_W = video_tensor.shape[2:]
    grid_pts = get_points_on_a_grid(grid_size, (frame_H, frame_W), device="cpu")
    
    if os.path.exists(mask_dir):
        grid_pts_int = grid_pts[0].long()
        mask_values = mask[grid_pts_int[...,1], grid_pts_int[...,0]]
        grid_pts = grid_pts[:, mask_values]
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
            # resize if scale is less than 1
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
        if viz:
            viser.visualize(video=video[None],
                            tracks=track2d_pred[None][...,:2],
                            visibility=vis_pred[None],filename="test")
        data_npz_load["coords"] = (torch.einsum("tij,tnj->tni", c2w_traj[:,:3,:3], track3d_pred[:,:,:3].cpu()) + c2w_traj[:,:3,3][:,None,:]).numpy()
        data_npz_load["extrinsics"] = torch.inverse(c2w_traj).cpu().numpy()
        data_npz_load["intrinsics"] = intrs.cpu().numpy()
        depth_save = point_map[:,2,...]
        depth_save[conf_depth<0.5] = 0
        data_npz_load["depths"] = depth_save.cpu().numpy()
        data_npz_load["video"] = (video_tensor).cpu().numpy()/255
        data_npz_load["visibs"] = vis_pred.cpu().numpy()
        data_npz_load["unc_metric"] = conf_depth.cpu().numpy()
        np.savez(os.path.join(out_dir, f'result.npz'), **data_npz_load)
        print(f"Results saved to {out_dir}.\nTo visualize them with tapip3d, run: [bold yellow]python tapip3d_viz.py {out_dir}/result.npz[/bold yellow]")
