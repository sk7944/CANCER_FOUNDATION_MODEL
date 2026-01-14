"""
Feature Extraction Module for WSI Preprocessing
================================================

Extracts feature vectors from image patches using pretrained models.

Supported models:
- ResNet-50 (ImageNet pretrained)
- Swin-T/S/B (ImageNet pretrained)
- Custom models (e.g., CTransPath, UNI - if available)

Output:
- Feature vectors stored in HDF5 format for efficient loading during training
"""

import os
import numpy as np
from typing import List, Optional, Union, Tuple, Dict, Any
from dataclasses import dataclass
import h5py
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not available. Install with: pip install timm")


@dataclass
class FeatureExtractionResult:
    """Result of feature extraction for a single WSI."""
    features: np.ndarray  # (N, D) feature matrix
    coordinates: np.ndarray  # (N, 2) patch coordinates
    slide_id: str
    model_name: str
    feature_dim: int


class PatchDataset(Dataset):
    """Simple dataset for batch processing of patches."""

    def __init__(
        self,
        patches: List[np.ndarray],
        coordinates: np.ndarray,
        transform: Optional[transforms.Compose] = None,
    ):
        self.patches = patches
        self.coordinates = coordinates
        self.transform = transform

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, np.ndarray]:
        image = self.patches[idx]

        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert to tensor and normalize
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, self.coordinates[idx]


class FeatureExtractor:
    """
    Extract features from image patches using pretrained models.

    Args:
        model_name: Name of the model to use. Options:
            - 'resnet50': ResNet-50 pretrained on ImageNet
            - 'swin_tiny': Swin-T pretrained on ImageNet
            - 'swin_small': Swin-S pretrained on ImageNet
            - 'swin_base': Swin-B pretrained on ImageNet
        device: Device to run inference on ('cuda' or 'cpu')
        batch_size: Batch size for inference
        num_workers: Number of data loading workers
        normalize: Whether to apply ImageNet normalization
    """

    # Model configurations
    MODEL_CONFIGS = {
        'resnet50': {
            'timm_name': 'resnet50',
            'feature_dim': 2048,
            'input_size': 224,
        },
        'resnet101': {
            'timm_name': 'resnet101',
            'feature_dim': 2048,
            'input_size': 224,
        },
        'swin_tiny': {
            'timm_name': 'swin_tiny_patch4_window7_224',
            'feature_dim': 768,
            'input_size': 224,
        },
        'swin_small': {
            'timm_name': 'swin_small_patch4_window7_224',
            'feature_dim': 768,
            'input_size': 224,
        },
        'swin_base': {
            'timm_name': 'swin_base_patch4_window7_224',
            'feature_dim': 1024,
            'input_size': 224,
        },
        'vit_base': {
            'timm_name': 'vit_base_patch16_224',
            'feature_dim': 768,
            'input_size': 224,
        },
        'convnext_tiny': {
            'timm_name': 'convnext_tiny',
            'feature_dim': 768,
            'input_size': 224,
        },
        'convnext_small': {
            'timm_name': 'convnext_small',
            'feature_dim': 768,
            'input_size': 224,
        },
    }

    # ImageNet normalization
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        model_name: str = 'resnet50',
        device: Optional[str] = None,
        batch_size: int = 64,
        num_workers: int = 4,
        normalize: bool = True,
    ):
        if not TIMM_AVAILABLE:
            raise ImportError(
                "timm is required for feature extraction. "
                "Install with: pip install timm"
            )

        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available: {list(self.MODEL_CONFIGS.keys())}"
            )

        self.model_name = model_name
        self.config = self.MODEL_CONFIGS[model_name]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalize = normalize

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Load model
        self.model = self._load_model()
        self.model.eval()

        # Build transform
        self.transform = self._build_transform()

        print(f"FeatureExtractor initialized:")
        print(f"  Model: {model_name}")
        print(f"  Feature dim: {self.config['feature_dim']}")
        print(f"  Device: {self.device}")

    def _load_model(self) -> nn.Module:
        """Load pretrained model and remove classification head."""
        model = timm.create_model(
            self.config['timm_name'],
            pretrained=True,
            num_classes=0,  # Remove classification head, output features
        )
        model = model.to(self.device)
        return model

    def _build_transform(self) -> transforms.Compose:
        """Build image transformation pipeline."""
        transform_list = [
            transforms.ToPILImage(),
            transforms.Resize(self.config['input_size']),
            transforms.ToTensor(),
        ]

        if self.normalize:
            transform_list.append(
                transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
            )

        return transforms.Compose(transform_list)

    @torch.no_grad()
    def extract_features(
        self,
        patches: List[np.ndarray],
        coordinates: np.ndarray,
        show_progress: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from a list of patches.

        Args:
            patches: List of RGB images (H, W, 3), uint8
            coordinates: Array of (x, y) coordinates, shape (N, 2)
            show_progress: Whether to show progress bar

        Returns:
            features: (N, D) feature matrix
            coordinates: (N, 2) coordinate matrix
        """
        if len(patches) == 0:
            return np.empty((0, self.config['feature_dim'])), np.empty((0, 2))

        # Create dataset and dataloader
        dataset = PatchDataset(patches, coordinates, self.transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if self.device.type == 'cuda' else False,
        )

        all_features = []
        all_coords = []

        iterator = tqdm(dataloader, desc="Extracting features") if show_progress else dataloader

        for batch_images, batch_coords in iterator:
            batch_images = batch_images.to(self.device)

            # Extract features
            features = self.model(batch_images)

            all_features.append(features.cpu().numpy())
            all_coords.append(batch_coords.numpy())

        # Concatenate results
        features = np.concatenate(all_features, axis=0)
        coordinates = np.concatenate(all_coords, axis=0)

        return features, coordinates

    @torch.no_grad()
    def extract_single(self, patch: np.ndarray) -> np.ndarray:
        """
        Extract features from a single patch.

        Args:
            patch: RGB image (H, W, 3), uint8

        Returns:
            Feature vector (D,)
        """
        # Apply transform
        image = self.transform(patch).unsqueeze(0).to(self.device)

        # Extract features
        features = self.model(image)

        return features.cpu().numpy().squeeze()

    @property
    def feature_dim(self) -> int:
        """Get the feature dimension of the model."""
        return self.config['feature_dim']


class FeatureStorage:
    """
    Storage utilities for extracted features.

    Features are stored in HDF5 format with the following structure:
        /slide_id/
            features: (N, D) float32
            coordinates: (N, 2) int64
            attrs:
                model_name: str
                feature_dim: int
                num_patches: int
    """

    @staticmethod
    def save_features(
        output_path: str,
        slide_id: str,
        features: np.ndarray,
        coordinates: np.ndarray,
        model_name: str,
        overwrite: bool = False,
    ) -> None:
        """
        Save features to HDF5 file.

        Args:
            output_path: Path to HDF5 file
            slide_id: Unique identifier for the slide
            features: (N, D) feature matrix
            coordinates: (N, 2) coordinate matrix
            model_name: Name of the model used
            overwrite: Whether to overwrite existing data
        """
        mode = 'w' if not os.path.exists(output_path) else 'a'

        with h5py.File(output_path, mode) as f:
            if slide_id in f:
                if overwrite:
                    del f[slide_id]
                else:
                    raise ValueError(f"Slide {slide_id} already exists. Use overwrite=True.")

            grp = f.create_group(slide_id)
            grp.create_dataset('features', data=features.astype(np.float32), compression='gzip')
            grp.create_dataset('coordinates', data=coordinates.astype(np.int64), compression='gzip')

            grp.attrs['model_name'] = model_name
            grp.attrs['feature_dim'] = features.shape[1] if features.ndim > 1 else 0
            grp.attrs['num_patches'] = len(features)

    @staticmethod
    def load_features(
        input_path: str,
        slide_id: str,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Load features from HDF5 file.

        Args:
            input_path: Path to HDF5 file
            slide_id: Unique identifier for the slide

        Returns:
            features: (N, D) feature matrix
            coordinates: (N, 2) coordinate matrix
            metadata: Dictionary of attributes
        """
        with h5py.File(input_path, 'r') as f:
            if slide_id not in f:
                raise KeyError(f"Slide {slide_id} not found in {input_path}")

            grp = f[slide_id]
            features = grp['features'][:]
            coordinates = grp['coordinates'][:]

            metadata = dict(grp.attrs)

        return features, coordinates, metadata

    @staticmethod
    def list_slides(input_path: str) -> List[str]:
        """List all slide IDs in an HDF5 file."""
        with h5py.File(input_path, 'r') as f:
            return list(f.keys())

    @staticmethod
    def save_single_slide(
        output_dir: str,
        slide_id: str,
        features: np.ndarray,
        coordinates: np.ndarray,
        model_name: str,
    ) -> str:
        """
        Save features for a single slide to its own HDF5 file.

        Args:
            output_dir: Directory to save the file
            slide_id: Unique identifier for the slide
            features: (N, D) feature matrix
            coordinates: (N, 2) coordinate matrix
            model_name: Name of the model used

        Returns:
            Path to the saved file
        """
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{slide_id}.h5")

        with h5py.File(output_path, 'w') as f:
            f.create_dataset('features', data=features.astype(np.float32), compression='gzip')
            f.create_dataset('coordinates', data=coordinates.astype(np.int64), compression='gzip')

            f.attrs['slide_id'] = slide_id
            f.attrs['model_name'] = model_name
            f.attrs['feature_dim'] = features.shape[1] if features.ndim > 1 else 0
            f.attrs['num_patches'] = len(features)

        return output_path


if __name__ == "__main__":
    print("FeatureExtractor module loaded successfully.")
    print(f"TIMM available: {TIMM_AVAILABLE}")

    if TIMM_AVAILABLE:
        print("\nAvailable models:")
        for name, config in FeatureExtractor.MODEL_CONFIGS.items():
            print(f"  {name}: {config['feature_dim']}-dim features")
