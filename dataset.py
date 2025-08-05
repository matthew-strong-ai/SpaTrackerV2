import os
import cv2
import numpy as np
import torch

class SpatialTrackerDataset:
    def __init__(self, data_dir, video_names=None, mask_ext=".png", video_ext=".mp4"):
        self.data_dir = data_dir
        self.mask_ext = mask_ext
        self.video_ext = video_ext
        if video_names is None:
            self.video_names = self._find_videos()
        else:
            self.video_names = video_names

    def _find_videos(self):
        # Find all video files in the directory
        files = os.listdir(self.data_dir)
        video_names = [f[:-len(self.video_ext)] for f in files if f.endswith(self.video_ext)]
        return video_names

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        video_path = os.path.join(self.data_dir, video_name + self.video_ext)
        mask_path = os.path.join(self.data_dir, video_name + self.mask_ext)
        video = self._read_video(video_path)
        mask = self._read_mask(mask_path, video.shape[1:3]) if os.path.exists(mask_path) else None
        return {
            "video_name": video_name,
            "video": video,
            "mask": mask
        }

    def _read_video(self, video_path, max_frames=100, fps=1):
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
        if not frames:
            raise RuntimeError(f"No frames read from {video_path}")
        video_tensor = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float()
        video_tensor = video_tensor[::fps].float()[:max_frames]
        return video_tensor

    def _read_mask(self, mask_path, shape):
        mask = cv2.imread(mask_path)
        mask = cv2.resize(mask, (shape[1], shape[0]))
        mask = mask.sum(axis=-1) > 0
        return mask
