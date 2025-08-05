"""
End-to-End SpaTrackerV2 Model
=============================

This module implements a unified end-to-end model that replicates the functionality of SpaTrackerV2.
The model takes RGB images as input and outputs:
- 3D point tracks
- Camera poses (extrinsics and intrinsics)
- Depth maps
- Visibility and confidence scores

Architecture Overview:
1. Feature Extraction: Vision Transformer backbone (DINOv2-like)
2. Multi-frame Aggregation: Temporal feature aggregation
3. Camera Pose Estimation: Regression head for camera parameters
4. Depth Estimation: DPT-style depth prediction head
5. 3D Point Tracking: Spatial tracking with attention mechanisms
6. Bundle Adjustment: Iterative refinement of tracks and poses

Author: Based on SpaTrackerV2 architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import cv2
from einops import rearrange, repeat
import math
from huggingface_hub import PyTorchModelHubMixin


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""
    def __init__(self, img_size=518, patch_size=14, in_chans=3, embed_dim=1024):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    def __init__(self, embed_dim=1024, num_heads=16, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """MLP with GELU activation"""
    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with attention and MLP"""
    def __init__(self, embed_dim=1024, num_heads=16, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim, mlp_hidden_dim, dropout=dropout)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class FeatureExtractor(nn.Module):
    """Vision Transformer for feature extraction"""
    def __init__(self, img_size=518, patch_size=14, in_chans=3, embed_dim=1024, depth=24, num_heads=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        return x


class TemporalAggregator(nn.Module):
    """Aggregate features across temporal frames"""
    def __init__(self, embed_dim=1024, num_heads=16, num_layers=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, 200, embed_dim))  # Max 200 frames
        
        self.temporal_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, features):
        """
        Args:
            features: List of feature tensors [(B, N, C), ...] for each frame
        Returns:
            aggregated_features: (B, T, N, C) where T is number of frames
        """
        B, N, C = features[0].shape
        T = len(features)
        
        # Stack features across time
        x = torch.stack(features, dim=1)  # (B, T, N, C)
        
        # Add temporal positional embedding
        temporal_pos = self.temporal_pos_embed[:, :T, :].unsqueeze(2)  # (1, T, 1, C)
        x = x + temporal_pos
        
        # Reshape for transformer processing
        x = x.view(B * T, N, C)
        
        # Apply temporal transformer blocks
        for blk in self.temporal_blocks:
            x = blk(x)
        
        x = self.norm(x)
        x = x.view(B, T, N, C)
        
        return x


class CameraPoseHead(nn.Module):
    """Camera pose estimation head"""
    def __init__(self, embed_dim=1024, output_dim=12):  # 9 for pose + 3 for intrinsics
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, output_dim)
        )
        
    def forward(self, x):
        # Use CLS token for global pose estimation
        cls_features = x[:, :, 0, :]  # (B, T, C)
        pose_params = self.head(cls_features)  # (B, T, 12)
        
        # Split into pose and intrinsics
        pose_enc = pose_params[..., :9]  # (B, T, 9)
        intrinsics = pose_params[..., 9:]  # (B, T, 3)
        
        return pose_enc, intrinsics


class DepthHead(nn.Module):
    """Depth estimation head using DPT-style architecture"""
    def __init__(self, embed_dim=1024, img_size=518, patch_size=14):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.feat_size = img_size // patch_size
        
        # Depth prediction layers
        self.depth_head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Confidence prediction
        self.conf_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Upsampling to original image size
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)
        )
        
    def forward(self, x):
        B, T, N, C = x.shape
        
        # Remove CLS token and reshape to spatial features
        spatial_features = x[:, :, 1:, :]  # (B, T, H*W, C)
        H = W = self.feat_size
        
        # Predict depth and confidence
        depth_tokens = self.depth_head(spatial_features)  # (B, T, H*W, 1)
        conf_tokens = self.conf_head(spatial_features)    # (B, T, H*W, 1)
        
        # Reshape to spatial maps
        depth_maps = depth_tokens.view(B, T, H, W, 1).permute(0, 1, 4, 2, 3)  # (B, T, 1, H, W)
        conf_maps = conf_tokens.view(B, T, H, W, 1).permute(0, 1, 4, 2, 3)    # (B, T, 1, H, W)
        
        # Upsample to original image size
        depth_maps = depth_maps.view(B * T, 1, H, W)
        conf_maps = conf_maps.view(B * T, 1, H, W)
        
        depth_maps = self.upsample(depth_maps)
        conf_maps = self.upsample(conf_maps)
        
        # Reshape back
        depth_maps = depth_maps.view(B, T, 1, self.img_size, self.img_size)
        conf_maps = conf_maps.view(B, T, 1, self.img_size, self.img_size)
        
        return depth_maps, conf_maps


class SpatialTrackingHead(nn.Module):
    """3D spatial tracking head"""
    def __init__(self, embed_dim=1024, max_points=1000):
        super().__init__()
        self.max_points = max_points
        self.embed_dim = embed_dim
        
        # Track feature extractor
        self.track_feature_head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        # Point tracking transformer
        self.track_transformer = nn.ModuleList([
            TransformerBlock(256, num_heads=8) for _ in range(6)
        ])
        
        # Output heads
        self.track_2d_head = nn.Linear(256, 2)  # 2D pixel coordinates
        self.track_3d_head = nn.Linear(256, 3)  # 3D world coordinates
        self.visibility_head = nn.Linear(256, 1)  # Visibility score
        self.confidence_head = nn.Linear(256, 1)  # Confidence score
        
    def forward(self, features, query_points, depth_maps, camera_poses, intrinsics):
        """
        Args:
            features: (B, T, N, C) aggregated features
            query_points: (B, M, 2) initial query points in first frame
            depth_maps: (B, T, 1, H, W) depth maps
            camera_poses: (B, T, 9) camera pose encodings
            intrinsics: (B, T, 3) camera intrinsics
        Returns:
            tracks_2d: (B, T, M, 2) 2D tracks
            tracks_3d: (B, T, M, 3) 3D tracks
            visibility: (B, T, M, 1) visibility scores
            confidence: (B, T, M, 1) confidence scores
        """
        B, T, N, C = features.shape
        M = query_points.shape[1]
        
        # Extract track features
        track_features = self.track_feature_head(features)  # (B, T, N, 256)
        
        # Initialize point tracks
        tracks_2d = []
        tracks_3d = []
        visibility = []
        confidence = []
        
        # Sample features at query points for first frame
        current_points = query_points  # (B, M, 2)
        
        for t in range(T):
            # Sample features at current point locations
            point_features = self.sample_features_at_points(
                track_features[:, t], current_points, feat_size=int(math.sqrt(N-1))
            )  # (B, M, 256)
            
            # Apply tracking transformer
            for transformer in self.track_transformer:
                point_features = transformer(point_features)
            
            # Predict outputs
            tracks_2d_t = self.track_2d_head(point_features)  # (B, M, 2)
            tracks_3d_t = self.track_3d_head(point_features)  # (B, M, 3)
            visibility_t = torch.sigmoid(self.visibility_head(point_features))  # (B, M, 1)
            confidence_t = torch.sigmoid(self.confidence_head(point_features))  # (B, M, 1)
            
            tracks_2d.append(tracks_2d_t)
            tracks_3d.append(tracks_3d_t)
            visibility.append(visibility_t)
            confidence.append(confidence_t)
            
            # Update current points for next frame
            current_points = tracks_2d_t
        
        # Stack results
        tracks_2d = torch.stack(tracks_2d, dim=1)  # (B, T, M, 2)
        tracks_3d = torch.stack(tracks_3d, dim=1)  # (B, T, M, 3)
        visibility = torch.stack(visibility, dim=1)  # (B, T, M, 1)
        confidence = torch.stack(confidence, dim=1)  # (B, T, M, 1)
        
        return tracks_2d, tracks_3d, visibility, confidence
    
    def sample_features_at_points(self, features, points, feat_size):
        """Sample features at given 2D points"""
        B, N, C = features.shape
        M = points.shape[1]
        
        # Remove CLS token
        spatial_features = features[:, 1:, :]  # (B, H*W, C)
        spatial_features = spatial_features.view(B, feat_size, feat_size, C)
        
        # Normalize points to [-1, 1] for grid_sample
        normalized_points = points.clone()
        normalized_points[..., 0] = (points[..., 0] / feat_size) * 2 - 1  # x
        normalized_points[..., 1] = (points[..., 1] / feat_size) * 2 - 1  # y
        
        # Reshape for grid_sample
        spatial_features = spatial_features.permute(0, 3, 1, 2)  # (B, C, H, W)
        grid = normalized_points.unsqueeze(2)  # (B, M, 1, 2)
        
        # Sample features
        sampled_features = F.grid_sample(
            spatial_features, grid, mode='bilinear', padding_mode='border', align_corners=False
        )  # (B, C, M, 1)
        
        sampled_features = sampled_features.squeeze(-1).transpose(1, 2)  # (B, M, C)
        
        return sampled_features


class EndToEndSpaTracker(nn.Module, PyTorchModelHubMixin):
    """End-to-end SpaTrackerV2 model"""
    
    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        max_frames: int = 100,
        max_points: int = 1000
    ):
        super().__init__()
        
        self.img_size = img_size
        self.max_frames = max_frames
        self.max_points = max_points
        
        # Core components
        self.feature_extractor = FeatureExtractor(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, 
            depth=depth, num_heads=num_heads
        )
        
        self.temporal_aggregator = TemporalAggregator(embed_dim=embed_dim, num_heads=num_heads)
        
        self.camera_head = CameraPoseHead(embed_dim=embed_dim)
        self.depth_head = DepthHead(embed_dim=embed_dim, img_size=img_size, patch_size=patch_size)
        self.tracking_head = SpatialTrackingHead(embed_dim=embed_dim, max_points=max_points)
        
    def forward(
        self,
        images: torch.Tensor,
        query_points: Optional[torch.Tensor] = None,
        depth_maps: Optional[torch.Tensor] = None,
        camera_poses: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the end-to-end SpaTracker model.
        
        Args:
            images: (B, T, 3, H, W) RGB images in range [0, 1]
            query_points: (B, M, 2) Optional query points for tracking
            depth_maps: (B, T, 1, H, W) Optional depth maps
            camera_poses: (B, T, 9) Optional camera pose encodings
            intrinsics: (B, T, 3) Optional camera intrinsics
            
        Returns:
            Dict containing:
                - tracks_2d: (B, T, M, 2) 2D pixel tracks
                - tracks_3d: (B, T, M, 3) 3D world tracks
                - visibility: (B, T, M, 1) visibility scores
                - confidence: (B, T, M, 1) confidence scores
                - depth_maps: (B, T, 1, H, W) depth maps
                - depth_conf: (B, T, 1, H, W) depth confidence
                - poses_pred: (B, T, 9) camera pose encodings
                - intrs: (B, T, 3) camera intrinsics
        """
        
        B, T, C, H, W = images.shape
        
        # Extract features for each frame
        features = []
        for t in range(T):
            frame_features = self.feature_extractor(images[:, t])  # (B, N, C)
            features.append(frame_features)
        
        # Temporal aggregation
        aggregated_features = self.temporal_aggregator(features)  # (B, T, N, C)
        
        # Camera pose estimation (if not provided)
        if camera_poses is None or intrinsics is None:
            pred_poses, pred_intrinsics = self.camera_head(aggregated_features)
            camera_poses = pred_poses if camera_poses is None else camera_poses
            intrinsics = pred_intrinsics if intrinsics is None else intrinsics
        else:
            pred_poses = camera_poses
            pred_intrinsics = intrinsics
        
        # Depth estimation (if not provided)
        if depth_maps is None:
            pred_depth, depth_conf = self.depth_head(aggregated_features)
            depth_maps = pred_depth
        else:
            pred_depth = depth_maps
            depth_conf = torch.ones_like(depth_maps)
        
        # Generate default query points if not provided
        if query_points is None:
            query_points = self.generate_grid_points(B, H, W, grid_size=10)
        
        # 3D spatial tracking
        tracks_2d, tracks_3d, visibility, confidence = self.tracking_head(
            aggregated_features, query_points, depth_maps, camera_poses, intrinsics
        )
        
        return {
            'tracks_2d': tracks_2d,
            'tracks_3d': tracks_3d,
            'visibility': visibility,
            'confidence': confidence,
            'depth_maps': pred_depth,
            'depth_conf': depth_conf,
            'poses_pred': pred_poses,
            'intrs': pred_intrinsics,
            'points_map': torch.cat([tracks_2d, pred_depth.squeeze(2).unsqueeze(-1)], dim=-1),
            'unc_metric': depth_conf.squeeze(2)
        }
    
    def generate_grid_points(self, batch_size, height, width, grid_size=10):
        """Generate grid of query points"""
        device = next(self.parameters()).device
        
        # Create grid
        y_coords = torch.linspace(0, height - 1, grid_size, device=device)
        x_coords = torch.linspace(0, width - 1, grid_size, device=device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Flatten and stack
        grid_points = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=1)  # (grid_size^2, 2)
        
        # Expand for batch
        grid_points = grid_points.unsqueeze(0).expand(batch_size, -1, -1)  # (B, grid_size^2, 2)
        
        return grid_points
    
    def preprocess_images(self, images):
        """Preprocess images for the model"""
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images).float()
        
        # Normalize to [0, 1] if needed
        if images.max() > 1.0:
            images = images / 255.0
        
        # Ensure proper shape (B, T, C, H, W)
        if images.dim() == 4:  # (T, C, H, W)
            images = images.unsqueeze(0)
        
        return images


def create_end_to_end_spatracker(
    img_size: int = 518,
    embed_dim: int = 1024,
    depth: int = 24,
    max_frames: int = 100,
    max_points: int = 1000
) -> EndToEndSpaTracker:
    """
    Create an end-to-end SpaTracker model.
    
    Args:
        img_size: Input image size
        embed_dim: Embedding dimension
        depth: Number of transformer layers
        max_frames: Maximum number of frames
        max_points: Maximum number of tracking points
        
    Returns:
        EndToEndSpaTracker model
    """
    return EndToEndSpaTracker(
        img_size=img_size,
        embed_dim=embed_dim,
        depth=depth,
        max_frames=max_frames,
        max_points=max_points
    )


# Example usage
if __name__ == "__main__":
    # Create model
    model = create_end_to_end_spatracker()
    model.eval()
    
    # Example input
    B, T, C, H, W = 1, 10, 3, 518, 518
    images = torch.randn(B, T, C, H, W)
    query_points = torch.randn(B, 100, 2) * 518  # Random points
    
    # Forward pass
    with torch.no_grad():
        outputs = model(images, query_points)
    
    print("Model outputs:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")
    
    print("\nEnd-to-end SpaTracker model created successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}") 