"""
Tissue Detection Module for WSI Preprocessing
==============================================

Detects tissue regions in WSI and generates binary masks to filter out
background (white/glass) regions.

Methods:
- Otsu thresholding on grayscale/saturation channel
- Morphological operations for noise removal
- Connected component analysis for region filtering
"""

import numpy as np
import cv2
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class TissueDetectionResult:
    """Result of tissue detection."""
    mask: np.ndarray  # Binary mask (H, W), True = tissue
    tissue_ratio: float  # Ratio of tissue to total area
    thumbnail: np.ndarray  # Downsampled RGB image used for detection
    contours: list  # List of tissue contours


class TissueDetector:
    """
    Detect tissue regions in Whole Slide Images.

    Uses HSV color space and Otsu thresholding to separate tissue from background.

    Args:
        saturation_threshold: Minimum saturation value for tissue (0-255)
        tissue_min_area: Minimum area (in pixels at thumbnail scale) for valid tissue region
        morph_kernel_size: Kernel size for morphological operations
        thumbnail_size: Size of thumbnail for processing (longer edge)
    """

    def __init__(
        self,
        saturation_threshold: Optional[int] = None,  # None = use Otsu
        tissue_min_area: int = 1000,
        morph_kernel_size: int = 7,
        thumbnail_size: int = 2048,
    ):
        self.saturation_threshold = saturation_threshold
        self.tissue_min_area = tissue_min_area
        self.morph_kernel_size = morph_kernel_size
        self.thumbnail_size = thumbnail_size

    def detect(
        self,
        image: np.ndarray,
        return_contours: bool = False,
    ) -> TissueDetectionResult:
        """
        Detect tissue regions in an image.

        Args:
            image: RGB image (H, W, 3) - can be full resolution or thumbnail
            return_contours: Whether to compute and return contours

        Returns:
            TissueDetectionResult with mask and statistics
        """
        # Ensure RGB format
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected RGB image (H, W, 3), got shape {image.shape}")

        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1]

        # Threshold saturation channel
        if self.saturation_threshold is not None:
            threshold = self.saturation_threshold
        else:
            # Use Otsu's method for automatic thresholding
            threshold, _ = cv2.threshold(
                saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

        # Create binary mask
        mask = saturation > threshold

        # Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morph_kernel_size, self.morph_kernel_size)
        )

        # Close small holes
        mask_cleaned = cv2.morphologyEx(
            mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel
        )

        # Open to remove small noise
        mask_cleaned = cv2.morphologyEx(
            mask_cleaned, cv2.MORPH_OPEN, kernel
        )

        # Remove small connected components
        mask_cleaned = self._remove_small_regions(mask_cleaned)

        # Convert back to boolean
        mask_final = mask_cleaned.astype(bool)

        # Calculate tissue ratio
        tissue_ratio = np.sum(mask_final) / mask_final.size

        # Get contours if requested
        contours = []
        if return_contours:
            contours, _ = cv2.findContours(
                mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contours = [c for c in contours if cv2.contourArea(c) >= self.tissue_min_area]

        return TissueDetectionResult(
            mask=mask_final,
            tissue_ratio=tissue_ratio,
            thumbnail=image,
            contours=contours,
        )

    def _remove_small_regions(self, mask: np.ndarray) -> np.ndarray:
        """Remove connected components smaller than tissue_min_area."""
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        # Create output mask
        output = np.zeros_like(mask)

        # Keep only large components (skip label 0 which is background)
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= self.tissue_min_area:
                output[labels == label] = 1

        return output

    def get_tissue_coordinates(
        self,
        mask: np.ndarray,
        patch_size: int,
        stride: Optional[int] = None,
        min_tissue_ratio: float = 0.8,
        scale_factor: float = 1.0,
    ) -> np.ndarray:
        """
        Get coordinates of patches that contain sufficient tissue.

        Args:
            mask: Binary tissue mask
            patch_size: Size of patches at full resolution
            stride: Stride between patches (default: patch_size, no overlap)
            min_tissue_ratio: Minimum ratio of tissue in patch (0-1)
            scale_factor: Scale factor from mask to full resolution

        Returns:
            Array of (x, y) coordinates at full resolution, shape (N, 2)
        """
        if stride is None:
            stride = patch_size

        # Convert to mask scale
        mask_patch_size = int(patch_size / scale_factor)
        mask_stride = int(stride / scale_factor)

        if mask_patch_size < 1:
            mask_patch_size = 1
        if mask_stride < 1:
            mask_stride = 1

        h, w = mask.shape
        coordinates = []

        for y in range(0, h - mask_patch_size + 1, mask_stride):
            for x in range(0, w - mask_patch_size + 1, mask_stride):
                # Extract patch from mask
                patch_mask = mask[y:y + mask_patch_size, x:x + mask_patch_size]

                # Calculate tissue ratio
                tissue_ratio = np.mean(patch_mask)

                if tissue_ratio >= min_tissue_ratio:
                    # Convert to full resolution coordinates
                    full_x = int(x * scale_factor)
                    full_y = int(y * scale_factor)
                    coordinates.append([full_x, full_y])

        return np.array(coordinates, dtype=np.int64) if coordinates else np.empty((0, 2), dtype=np.int64)

    def visualize_detection(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.3,
    ) -> np.ndarray:
        """
        Visualize tissue detection result by overlaying mask on image.

        Args:
            image: RGB image
            mask: Binary tissue mask
            alpha: Transparency of overlay

        Returns:
            RGB image with tissue regions highlighted
        """
        # Create colored overlay
        overlay = image.copy()
        overlay[mask] = [0, 255, 0]  # Green for tissue

        # Blend with original
        result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

        return result


if __name__ == "__main__":
    # Test with a simple synthetic image
    print("TissueDetector module loaded successfully.")

    # Create test image (white background with colored region)
    test_image = np.ones((512, 512, 3), dtype=np.uint8) * 255
    test_image[100:400, 100:400] = [200, 150, 180]  # Simulated tissue (pinkish)

    detector = TissueDetector()
    result = detector.detect(test_image, return_contours=True)

    print(f"Tissue ratio: {result.tissue_ratio:.2%}")
    print(f"Mask shape: {result.mask.shape}")
    print(f"Number of contours: {len(result.contours)}")
