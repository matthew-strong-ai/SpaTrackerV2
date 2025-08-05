import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image
from models.SpaTrackV2.models.predictor import Predictor
from models.SpaTrackV2.utils.visualizer import Visualizer
from models.SpaTrackV2.models.utils import get_points_on_a_grid

class SpatialTrackerRunner:
    def __init__(self, 
                 track_mode="offline", 
                 data_type="RGB", 
                 data_dir="examples", 
                 video_name="snowboard", 
                 grid_size=10, 
                 vo_points=756, 
                 fps=1):
        self.track_mode = track_mode
        self.data_type = data_type
        self.data_dir = data_dir
        self.video_name = video_name
        self.grid_size = grid_size
        self.vo_points = vo_points
        self.fps = int(fps)
        self.out_dir = os.path.join(self.data_dir, "results")
        os.makedirs(self.out_dir, exist_ok=True)
        self.mask_dir = os.path.join(self.data_dir, f"{self.video_name}.png")
        self.video_path = os.path.join(self.data_dir, f"{self.video_name}.mp4")

    def read_video_opencv(self, video_path, max_frames=100, fps=1):
        cap = cv2.VideoCapture(video_path)
        frames = []
        count = 0
        while cap.isOpened() and len(frames) < (max_frames * 10):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            count += 1
        cap.release()
        video_tensor = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float()
        video_tensor = video_tensor[::fps].float()[:max_frames]
        return video_tensor

    def run(self, max_frames=100):
        vggt4track_model = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front")
        vggt4track_model.eval()
        vggt4track_model = vggt4track_model.to("cuda")

        print("Reading video with OpenCV...")
        video_tensor = self.read_video_opencv(self.video_path, max_frames=max_frames, fps=self.fps)
        print(f"Video tensor shape: {video_tensor.shape}")

        print("Processing tensor")
        video_tensor = preprocess_image(video_tensor)[None]
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                predictions = vggt4track_model(video_tensor.cuda()/255)
                extrinsic, intrinsic = predictions["poses_pred"], predictions["intrs"]
                depth_map, depth_conf = predictions["points_map"][..., 2], predictions["unc_metric"]

        depth_tensor = depth_map.squeeze().cpu().numpy()
        extrs = extrinsic.squeeze().cpu().numpy()
        intrs = intrinsic.squeeze().cpu().numpy()
        video_tensor = video_tensor.squeeze()
        unc_metric = depth_conf.squeeze().cpu().numpy() > 0.5
        data_npz_load = {}

        if os.path.exists(self.mask_dir):
            mask = cv2.imread(self.mask_dir)
            mask = cv2.resize(mask, (video_tensor.shape[2], video_tensor.shape[1]))
            mask = mask.sum(axis=-1) > 0
        else:
            mask = np.ones_like(video_tensor[0,0].numpy()) > 0

        # Model selection
        if self.track_mode == "offline":
            model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Offline")
        else:
            model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Online")
        model.spatrack.track_num = self.vo_points
        model.eval()
        model.to("cuda")

        viser = Visualizer(save_dir=self.out_dir, grayscale=True, fps=10, pad_value=0, tracks_leave_trace=5)
        grid_size = self.grid_size
        frame_H, frame_W = video_tensor.shape[2:]
        grid_pts = get_points_on_a_grid(grid_size, (frame_H, frame_W), device="cpu")
        
        if os.path.exists(self.mask_dir):
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
            np.savez(os.path.join(self.out_dir, f'result.npz'), **data_npz_load)
            print(f"Results saved to {self.out_dir}.\nTo visualize them with tapip3d, run: python tapip3d_viz.py {self.out_dir}/result.npz")
