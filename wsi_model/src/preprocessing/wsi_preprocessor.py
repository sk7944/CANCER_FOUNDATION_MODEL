"""
WSI Preprocessing Pipeline
==========================

Main orchestrator for the complete WSI preprocessing pipeline.

Pipeline steps:
1. Load WSI and generate thumbnail
2. Detect tissue regions
3. Get patch coordinates from tissue mask
4. Extract patches at coordinates
5. Apply stain normalization/augmentation
6. Extract features using pretrained model
7. Save features to HDF5

Usage:
    from wsi_model.src.preprocessing import WSIPreprocessor

    preprocessor = WSIPreprocessor(
        patch_size=256,
        model_name='swin_tiny',
        output_dir='path/to/output',
        stain_normalize=True,
    )

    # Process single WSI
    result = preprocessor.process_wsi('path/to/slide.svs')

    # Process multiple WSIs
    preprocessor.process_directory('path/to/wsi_dir', pattern='*.svs')
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from tqdm import tqdm
import numpy as np

from .tissue_detector import TissueDetector
from .patch_extractor import PatchExtractor, OPENSLIDE_AVAILABLE
from .stain_normalizer import StainNormalizer, StainAugmentor
from .feature_extractor import FeatureExtractor, FeatureStorage, TIMM_AVAILABLE

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for WSI preprocessing."""
    # Patch extraction
    patch_size: int = 256
    patch_level: int = 0
    min_tissue_ratio: float = 0.8
    blur_threshold: float = 50.0

    # Tissue detection
    thumbnail_size: int = 2048
    tissue_min_area: int = 1000

    # Stain normalization
    stain_normalize: bool = True
    stain_augment: bool = False
    stain_augment_sigma_alpha: float = 0.2
    stain_augment_sigma_beta: float = 0.2

    # Feature extraction
    model_name: str = 'swin_tiny'
    batch_size: int = 64
    num_workers: int = 4

    # Output
    save_patches: bool = False  # Whether to save patch images (uses lots of disk)
    save_features: bool = True  # Whether to save feature vectors


@dataclass
class WSIProcessingResult:
    """Result of processing a single WSI."""
    slide_id: str
    slide_path: str
    success: bool
    error_message: Optional[str]

    # Statistics
    num_patches_total: int
    num_patches_valid: int
    tissue_ratio: float
    processing_time_seconds: float

    # Output paths
    features_path: Optional[str]
    patches_dir: Optional[str]


class WSIPreprocessor:
    """
    Complete WSI preprocessing pipeline.

    Args:
        output_dir: Directory to save outputs
        config: PreprocessingConfig object or None for defaults
        device: Device for feature extraction ('cuda' or 'cpu')
    """

    def __init__(
        self,
        output_dir: str,
        config: Optional[PreprocessingConfig] = None,
        device: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.config = config or PreprocessingConfig()
        self.device = device

        # Create output directories
        self.features_dir = self.output_dir / 'features'
        self.patches_dir = self.output_dir / 'patches'
        self.logs_dir = self.output_dir / 'logs'

        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        if self.config.save_patches:
            self.patches_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self._init_components()

        # Save config
        self._save_config()

        logger.info(f"WSIPreprocessor initialized")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  Patch size: {self.config.patch_size}")
        logger.info(f"  Model: {self.config.model_name}")
        logger.info(f"  Stain normalization: {self.config.stain_normalize}")

    def _init_components(self):
        """Initialize preprocessing components."""
        # Tissue detector
        self.tissue_detector = TissueDetector(
            tissue_min_area=self.config.tissue_min_area,
            thumbnail_size=self.config.thumbnail_size,
        )

        # Patch extractor
        if OPENSLIDE_AVAILABLE:
            self.patch_extractor = PatchExtractor(
                patch_size=self.config.patch_size,
                level=self.config.patch_level,
                blur_threshold=self.config.blur_threshold,
                tissue_threshold=self.config.min_tissue_ratio,
                num_workers=self.config.num_workers,
            )
        else:
            logger.warning("OpenSlide not available. Patch extraction will be limited.")
            self.patch_extractor = None

        # Stain normalizer
        if self.config.stain_normalize:
            self.stain_normalizer = StainNormalizer()
        else:
            self.stain_normalizer = None

        # Stain augmentor
        if self.config.stain_augment:
            self.stain_augmentor = StainAugmentor(
                sigma_alpha=self.config.stain_augment_sigma_alpha,
                sigma_beta=self.config.stain_augment_sigma_beta,
            )
        else:
            self.stain_augmentor = None

        # Feature extractor
        if self.config.save_features and TIMM_AVAILABLE:
            self.feature_extractor = FeatureExtractor(
                model_name=self.config.model_name,
                device=self.device,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
            )
        else:
            self.feature_extractor = None
            if self.config.save_features:
                logger.warning("TIMM not available. Feature extraction disabled.")

    def _save_config(self):
        """Save configuration to file."""
        config_path = self.output_dir / 'preprocessing_config.json'
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)

    def process_wsi(
        self,
        slide_path: str,
        slide_id: Optional[str] = None,
    ) -> WSIProcessingResult:
        """
        Process a single WSI through the complete pipeline.

        Args:
            slide_path: Path to the WSI file (.svs, .tiff, etc.)
            slide_id: Unique identifier for the slide. If None, derived from filename.

        Returns:
            WSIProcessingResult with statistics and output paths
        """
        import time
        start_time = time.time()

        # Derive slide_id from filename if not provided
        if slide_id is None:
            slide_id = Path(slide_path).stem

        logger.info(f"Processing: {slide_id}")

        try:
            # Step 1: Open slide and get thumbnail
            slide = self.patch_extractor.open_slide(slide_path)
            thumbnail, scale_factor = self.patch_extractor.get_thumbnail(
                slide, self.config.thumbnail_size
            )

            # Step 2: Detect tissue
            tissue_result = self.tissue_detector.detect(thumbnail)
            logger.info(f"  Tissue ratio: {tissue_result.tissue_ratio:.1%}")

            if tissue_result.tissue_ratio < 0.01:
                raise ValueError("Insufficient tissue detected (<1%)")

            # Step 3: Get patch coordinates
            coordinates = self.tissue_detector.get_tissue_coordinates(
                mask=tissue_result.mask,
                patch_size=self.config.patch_size,
                min_tissue_ratio=self.config.min_tissue_ratio,
                scale_factor=scale_factor,
            )
            logger.info(f"  Total patch candidates: {len(coordinates)}")

            if len(coordinates) == 0:
                raise ValueError("No valid patch coordinates found")

            # Step 4: Extract patches
            patches = []
            valid_coords = []

            for patch_info in tqdm(
                self.patch_extractor.extract_patches_at_coordinates(
                    slide, coordinates, validate=True
                ),
                total=len(coordinates),
                desc="  Extracting patches",
                leave=False,
            ):
                if patch_info.is_valid:
                    patch_image = patch_info.image

                    # Step 5: Apply stain normalization
                    if self.stain_normalizer is not None:
                        try:
                            patch_image = self.stain_normalizer.normalize(patch_image)
                        except Exception:
                            pass  # Keep original if normalization fails

                    patches.append(patch_image)
                    valid_coords.append([patch_info.x, patch_info.y])

            slide.close()

            valid_coords = np.array(valid_coords)
            logger.info(f"  Valid patches: {len(patches)}")

            if len(patches) == 0:
                raise ValueError("No patches passed quality filters")

            # Step 6: Extract features
            features_path = None
            if self.feature_extractor is not None and self.config.save_features:
                features, coords = self.feature_extractor.extract_features(
                    patches, valid_coords, show_progress=True
                )

                # Save features
                features_path = str(self.features_dir / f"{slide_id}.h5")
                FeatureStorage.save_single_slide(
                    output_dir=str(self.features_dir),
                    slide_id=slide_id,
                    features=features,
                    coordinates=coords,
                    model_name=self.config.model_name,
                )
                logger.info(f"  Features saved: {features_path}")

            # Step 7: Save patches (optional)
            patches_path = None
            if self.config.save_patches:
                patches_path = str(self.patches_dir / slide_id)
                self._save_patches(patches, valid_coords, patches_path)
                logger.info(f"  Patches saved: {patches_path}")

            processing_time = time.time() - start_time

            return WSIProcessingResult(
                slide_id=slide_id,
                slide_path=slide_path,
                success=True,
                error_message=None,
                num_patches_total=len(coordinates),
                num_patches_valid=len(patches),
                tissue_ratio=tissue_result.tissue_ratio,
                processing_time_seconds=processing_time,
                features_path=features_path,
                patches_dir=patches_path,
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"  Error: {str(e)}")

            return WSIProcessingResult(
                slide_id=slide_id,
                slide_path=slide_path,
                success=False,
                error_message=str(e),
                num_patches_total=0,
                num_patches_valid=0,
                tissue_ratio=0.0,
                processing_time_seconds=processing_time,
                features_path=None,
                patches_dir=None,
            )

    def _save_patches(
        self,
        patches: List[np.ndarray],
        coordinates: np.ndarray,
        output_dir: str,
    ):
        """Save extracted patches as images."""
        import cv2

        os.makedirs(output_dir, exist_ok=True)

        for i, (patch, coord) in enumerate(zip(patches, coordinates)):
            x, y = coord
            filename = f"patch_{x}_{y}.png"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))

    def process_directory(
        self,
        input_dir: str,
        pattern: str = "*.svs",
        recursive: bool = True,
        max_workers: int = 1,  # Serial processing by default for GPU
    ) -> List[WSIProcessingResult]:
        """
        Process all WSI files in a directory.

        Args:
            input_dir: Directory containing WSI files
            pattern: Glob pattern for WSI files
            recursive: Whether to search recursively
            max_workers: Number of parallel workers (1 for serial)

        Returns:
            List of WSIProcessingResult objects
        """
        from glob import glob

        # Find all WSI files
        if recursive:
            search_pattern = os.path.join(input_dir, "**", pattern)
        else:
            search_pattern = os.path.join(input_dir, pattern)

        slide_paths = glob(search_pattern, recursive=recursive)
        logger.info(f"Found {len(slide_paths)} WSI files")

        results = []

        for slide_path in tqdm(slide_paths, desc="Processing WSIs"):
            result = self.process_wsi(slide_path)
            results.append(result)

            # Save intermediate results
            self._save_results(results)

        return results

    def _save_results(self, results: List[WSIProcessingResult]):
        """Save processing results to JSON."""
        results_path = self.logs_dir / 'processing_results.json'

        results_dict = []
        for r in results:
            results_dict.append({
                'slide_id': r.slide_id,
                'slide_path': r.slide_path,
                'success': r.success,
                'error_message': r.error_message,
                'num_patches_total': r.num_patches_total,
                'num_patches_valid': r.num_patches_valid,
                'tissue_ratio': r.tissue_ratio,
                'processing_time_seconds': r.processing_time_seconds,
                'features_path': r.features_path,
            })

        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2)

    def get_summary(self, results: List[WSIProcessingResult]) -> Dict[str, Any]:
        """Get summary statistics from processing results."""
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        total_patches = sum(r.num_patches_valid for r in successful)
        total_time = sum(r.processing_time_seconds for r in results)

        return {
            'total_slides': len(results),
            'successful': len(successful),
            'failed': len(failed),
            'total_patches': total_patches,
            'avg_patches_per_slide': total_patches / len(successful) if successful else 0,
            'total_processing_time_minutes': total_time / 60,
            'avg_time_per_slide_seconds': total_time / len(results) if results else 0,
            'failed_slides': [r.slide_id for r in failed],
        }


def main():
    """Command-line interface for WSI preprocessing."""
    import argparse

    parser = argparse.ArgumentParser(description='WSI Preprocessing Pipeline')
    parser.add_argument('input', help='Input WSI file or directory')
    parser.add_argument('--output', '-o', required=True, help='Output directory')
    parser.add_argument('--patch-size', type=int, default=256, help='Patch size (default: 256)')
    parser.add_argument('--model', default='swin_tiny', help='Feature extraction model')
    parser.add_argument('--no-stain-norm', action='store_true', help='Disable stain normalization')
    parser.add_argument('--save-patches', action='store_true', help='Save patch images')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for feature extraction')
    parser.add_argument('--device', default=None, help='Device (cuda/cpu)')

    args = parser.parse_args()

    config = PreprocessingConfig(
        patch_size=args.patch_size,
        model_name=args.model,
        stain_normalize=not args.no_stain_norm,
        save_patches=args.save_patches,
        batch_size=args.batch_size,
    )

    preprocessor = WSIPreprocessor(
        output_dir=args.output,
        config=config,
        device=args.device,
    )

    if os.path.isfile(args.input):
        result = preprocessor.process_wsi(args.input)
        print(f"Result: {'Success' if result.success else 'Failed'}")
        print(f"Valid patches: {result.num_patches_valid}")
    else:
        results = preprocessor.process_directory(args.input)
        summary = preprocessor.get_summary(results)
        print(f"\nSummary:")
        print(f"  Processed: {summary['total_slides']} slides")
        print(f"  Successful: {summary['successful']}")
        print(f"  Failed: {summary['failed']}")
        print(f"  Total patches: {summary['total_patches']}")


if __name__ == "__main__":
    main()
