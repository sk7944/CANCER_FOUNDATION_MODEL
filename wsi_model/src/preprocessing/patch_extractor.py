"""
Patch Extraction Module for WSI Preprocessing
==============================================

Extracts patches from Whole Slide Images at specified locations.
Handles multi-resolution pyramid structure of SVS files.

Supports:
- OpenSlide for SVS/TIFF reading
- Multi-threaded extraction
- Quality filtering (blur detection)
"""

import os
import numpy as np
from typing import List, Tuple, Optional, Generator, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import cv2

try:
    import openslide
    OPENSLIDE_AVAILABLE = True
except ImportError:
    OPENSLIDE_AVAILABLE = False
    print("Warning: OpenSlide not available. Install with: pip install openslide-python")


@dataclass
class PatchInfo:
    """Information about an extracted patch."""
    image: np.ndarray  # RGB image (H, W, 3)
    x: int  # Top-left x coordinate at level 0
    y: int  # Top-left y coordinate at level 0
    level: int  # Pyramid level
    size: int  # Patch size
    is_valid: bool  # Whether patch passed quality checks
    blur_score: float  # Laplacian variance (higher = sharper)
    tissue_ratio: float  # Ratio of tissue pixels


class PatchExtractor:
    """
    Extract patches from Whole Slide Images.

    Args:
        patch_size: Size of patches to extract (default: 256)
        level: Pyramid level to extract from (0 = full resolution)
        stride: Stride between patches (default: patch_size, no overlap)
        blur_threshold: Minimum Laplacian variance for sharp patches
        tissue_threshold: Minimum tissue ratio in patch
        num_workers: Number of threads for parallel extraction
    """

    def __init__(
        self,
        patch_size: int = 256,
        level: int = 0,
        stride: Optional[int] = None,
        blur_threshold: float = 50.0,
        tissue_threshold: float = 0.8,
        num_workers: int = 4,
    ):
        if not OPENSLIDE_AVAILABLE:
            raise ImportError(
                "OpenSlide is required for patch extraction. "
                "Install with: pip install openslide-python"
            )

        self.patch_size = patch_size
        self.level = level
        self.stride = stride if stride is not None else patch_size
        self.blur_threshold = blur_threshold
        self.tissue_threshold = tissue_threshold
        self.num_workers = num_workers

    def open_slide(self, slide_path: str) -> 'openslide.OpenSlide':
        """
        Open a slide file.

        Args:
            slide_path: Path to SVS/TIFF file

        Returns:
            OpenSlide object
        """
        if not os.path.exists(slide_path):
            raise FileNotFoundError(f"Slide not found: {slide_path}")

        return openslide.OpenSlide(slide_path)

    def get_slide_info(self, slide: 'openslide.OpenSlide') -> dict:
        """
        Get information about a slide.

        Args:
            slide: OpenSlide object

        Returns:
            Dictionary with slide information
        """
        return {
            'dimensions': slide.dimensions,  # (width, height) at level 0
            'level_count': slide.level_count,
            'level_dimensions': slide.level_dimensions,
            'level_downsamples': slide.level_downsamples,
            'properties': dict(slide.properties),
        }

    def extract_patch(
        self,
        slide: 'openslide.OpenSlide',
        x: int,
        y: int,
        size: Optional[int] = None,
        level: Optional[int] = None,
    ) -> np.ndarray:
        """
        Extract a single patch from the slide.

        Args:
            slide: OpenSlide object
            x: Top-left x coordinate at level 0
            y: Top-left y coordinate at level 0
            size: Patch size (default: self.patch_size)
            level: Pyramid level (default: self.level)

        Returns:
            RGB image (H, W, 3), uint8
        """
        if size is None:
            size = self.patch_size
        if level is None:
            level = self.level

        # Read region returns RGBA
        region = slide.read_region((x, y), level, (size, size))

        # Convert to RGB numpy array
        image = np.array(region.convert('RGB'))

        return image

    def compute_blur_score(self, image: np.ndarray) -> float:
        """
        Compute blur score using Laplacian variance.
        Higher values indicate sharper images.

        Args:
            image: RGB image (H, W, 3)

        Returns:
            Blur score (Laplacian variance)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        return variance

    def compute_tissue_ratio(self, image: np.ndarray) -> float:
        """
        Compute ratio of tissue pixels in patch.

        Args:
            image: RGB image (H, W, 3)

        Returns:
            Tissue ratio (0-1)
        """
        # Convert to HSV and use saturation channel
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1]

        # Threshold to get tissue mask
        _, mask = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return np.mean(mask > 0)

    def is_valid_patch(
        self,
        image: np.ndarray,
        blur_threshold: Optional[float] = None,
        tissue_threshold: Optional[float] = None,
    ) -> Tuple[bool, float, float]:
        """
        Check if a patch passes quality filters.

        Args:
            image: RGB image (H, W, 3)
            blur_threshold: Minimum blur score
            tissue_threshold: Minimum tissue ratio

        Returns:
            (is_valid, blur_score, tissue_ratio)
        """
        if blur_threshold is None:
            blur_threshold = self.blur_threshold
        if tissue_threshold is None:
            tissue_threshold = self.tissue_threshold

        blur_score = self.compute_blur_score(image)
        tissue_ratio = self.compute_tissue_ratio(image)

        is_valid = (blur_score >= blur_threshold) and (tissue_ratio >= tissue_threshold)

        return is_valid, blur_score, tissue_ratio

    def extract_patches_at_coordinates(
        self,
        slide: 'openslide.OpenSlide',
        coordinates: np.ndarray,
        validate: bool = True,
    ) -> Generator[PatchInfo, None, None]:
        """
        Extract patches at specified coordinates.

        Args:
            slide: OpenSlide object
            coordinates: Array of (x, y) coordinates, shape (N, 2)
            validate: Whether to run quality checks

        Yields:
            PatchInfo for each coordinate
        """
        for x, y in coordinates:
            image = self.extract_patch(slide, int(x), int(y))

            if validate:
                is_valid, blur_score, tissue_ratio = self.is_valid_patch(image)
            else:
                is_valid = True
                blur_score = -1.0
                tissue_ratio = -1.0

            yield PatchInfo(
                image=image,
                x=int(x),
                y=int(y),
                level=self.level,
                size=self.patch_size,
                is_valid=is_valid,
                blur_score=blur_score,
                tissue_ratio=tissue_ratio,
            )

    def extract_patches_parallel(
        self,
        slide_path: str,
        coordinates: np.ndarray,
        validate: bool = True,
    ) -> List[PatchInfo]:
        """
        Extract patches in parallel using multiple threads.

        Args:
            slide_path: Path to slide file
            coordinates: Array of (x, y) coordinates, shape (N, 2)
            validate: Whether to run quality checks

        Returns:
            List of PatchInfo objects
        """
        def extract_single(coord):
            # Each thread needs its own slide object
            slide = self.open_slide(slide_path)
            x, y = coord

            try:
                image = self.extract_patch(slide, int(x), int(y))

                if validate:
                    is_valid, blur_score, tissue_ratio = self.is_valid_patch(image)
                else:
                    is_valid = True
                    blur_score = -1.0
                    tissue_ratio = -1.0

                return PatchInfo(
                    image=image,
                    x=int(x),
                    y=int(y),
                    level=self.level,
                    size=self.patch_size,
                    is_valid=is_valid,
                    blur_score=blur_score,
                    tissue_ratio=tissue_ratio,
                )
            finally:
                slide.close()

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(extract_single, coordinates))

        return results

    def get_thumbnail(
        self,
        slide: 'openslide.OpenSlide',
        size: int = 2048,
    ) -> Tuple[np.ndarray, float]:
        """
        Get a thumbnail of the slide.

        Args:
            slide: OpenSlide object
            size: Maximum dimension of thumbnail

        Returns:
            (thumbnail_image, scale_factor)
            scale_factor: ratio of level 0 to thumbnail
        """
        # Calculate thumbnail size maintaining aspect ratio
        w, h = slide.dimensions
        if w > h:
            thumb_w = size
            thumb_h = int(h * size / w)
        else:
            thumb_h = size
            thumb_w = int(w * size / h)

        # Get thumbnail
        thumbnail = slide.get_thumbnail((thumb_w, thumb_h))
        thumbnail = np.array(thumbnail.convert('RGB'))

        # Calculate scale factor
        scale_factor = w / thumb_w

        return thumbnail, scale_factor


class TiffPatchExtractor:
    """
    Fallback patch extractor using tifffile for when OpenSlide is not available.
    Limited functionality compared to OpenSlide version.
    """

    def __init__(
        self,
        patch_size: int = 256,
        level: int = 0,
    ):
        try:
            import tifffile
            self.tifffile = tifffile
        except ImportError:
            raise ImportError("tifffile is required. Install with: pip install tifffile")

        self.patch_size = patch_size
        self.level = level

    def extract_patch(
        self,
        tif_path: str,
        x: int,
        y: int,
    ) -> np.ndarray:
        """Extract a patch from a TIFF file."""
        with self.tifffile.TiffFile(tif_path) as tif:
            page = tif.pages[self.level]
            # Read region
            image = page.asarray()[
                y:y + self.patch_size,
                x:x + self.patch_size
            ]
            return image


if __name__ == "__main__":
    print("PatchExtractor module loaded successfully.")
    print(f"OpenSlide available: {OPENSLIDE_AVAILABLE}")
