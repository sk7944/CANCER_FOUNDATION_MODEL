"""
Stain Normalization and Augmentation Module
============================================

Implements Macenko stain normalization for H&E stained histopathology images.
Also provides stain augmentation for data augmentation during training.

Reference:
    Macenko, M., et al. "A method for normalizing histology slides for
    quantitative analysis." ISBI 2009.
"""

import numpy as np
from typing import Tuple, Optional
import cv2


class StainNormalizer:
    """
    Macenko Stain Normalization for H&E images.

    Separates the image into Hematoxylin and Eosin stain channels,
    then normalizes to a reference stain matrix.

    Args:
        target_stain_matrix: Target stain matrix (2, 3) for normalization.
                            If None, uses standard H&E reference.
        target_concentrations: Target stain concentrations (2,) for 99th percentile.
                              If None, uses standard values.
        luminosity_threshold: Threshold for background detection (0-1)
        regularizer: Small value to avoid log(0)
    """

    # Standard H&E stain vectors (Ruifrok and Johnston)
    REFERENCE_STAIN_MATRIX = np.array([
        [0.5626, 0.7201, 0.4062],  # Hematoxylin
        [0.2159, 0.8012, 0.5581],  # Eosin
    ])

    REFERENCE_CONCENTRATIONS = np.array([1.9705, 1.0308])

    def __init__(
        self,
        target_stain_matrix: Optional[np.ndarray] = None,
        target_concentrations: Optional[np.ndarray] = None,
        luminosity_threshold: float = 0.8,
        regularizer: float = 1e-6,
    ):
        self.target_stain_matrix = (
            target_stain_matrix if target_stain_matrix is not None
            else self.REFERENCE_STAIN_MATRIX
        )
        self.target_concentrations = (
            target_concentrations if target_concentrations is not None
            else self.REFERENCE_CONCENTRATIONS
        )
        self.luminosity_threshold = luminosity_threshold
        self.regularizer = regularizer

        # Precompute target inverse
        self._target_stain_matrix_inv = np.linalg.pinv(self.target_stain_matrix)

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize stain colors of an H&E image.

        Args:
            image: RGB image (H, W, 3), uint8 [0, 255]

        Returns:
            Normalized RGB image (H, W, 3), uint8 [0, 255]
        """
        # Convert to float and optical density
        image_float = image.astype(np.float64) / 255.0
        image_float = np.clip(image_float, self.regularizer, 1.0)

        # Get stain matrix and concentrations from source image
        try:
            source_stain_matrix, source_concentrations = self._get_stain_matrix(
                image_float
            )
        except Exception:
            # If stain estimation fails, return original image
            return image

        # Convert to optical density
        od = -np.log(image_float)
        od_flat = od.reshape(-1, 3)

        # Get source stain concentrations
        source_concentrations_image = od_flat @ np.linalg.pinv(source_stain_matrix).T

        # Normalize concentrations
        source_max = np.percentile(source_concentrations_image, 99, axis=0)
        source_max = np.clip(source_max, self.regularizer, None)

        normalized_concentrations = (
            source_concentrations_image
            * (self.target_concentrations / source_max)
        )

        # Reconstruct image with target stain matrix
        od_normalized = normalized_concentrations @ self.target_stain_matrix
        od_normalized = od_normalized.reshape(od.shape)

        # Convert back to RGB
        image_normalized = np.exp(-od_normalized)
        image_normalized = np.clip(image_normalized * 255, 0, 255).astype(np.uint8)

        return image_normalized

    def _get_stain_matrix(
        self,
        image_float: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate stain matrix from image using Macenko's method.

        Args:
            image_float: RGB image (H, W, 3), float [0, 1]

        Returns:
            stain_matrix: (2, 3) matrix of stain vectors
            concentrations: (2,) 99th percentile concentrations
        """
        # Convert to optical density
        od = -np.log(image_float + self.regularizer)
        od_flat = od.reshape(-1, 3)

        # Remove background pixels (low optical density)
        od_threshold = 0.15
        mask = np.all(od_flat > od_threshold, axis=1)

        if np.sum(mask) < 100:
            raise ValueError("Not enough tissue pixels for stain estimation")

        od_tissue = od_flat[mask]

        # SVD to find principal stain directions
        _, _, V = np.linalg.svd(od_tissue, full_matrices=False)

        # Project onto plane spanned by first two principal components
        plane = V[:2, :]
        projected = od_tissue @ plane.T

        # Find angle of each point
        angles = np.arctan2(projected[:, 1], projected[:, 0])

        # Find robust min and max angles (using percentiles)
        min_angle = np.percentile(angles, 1)
        max_angle = np.percentile(angles, 99)

        # Get stain vectors
        v1 = np.array([np.cos(min_angle), np.sin(min_angle)]) @ plane
        v2 = np.array([np.cos(max_angle), np.sin(max_angle)]) @ plane

        # Normalize
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)

        # Ensure H is first (typically has higher red component in OD space)
        if v1[0] < v2[0]:
            stain_matrix = np.array([v1, v2])
        else:
            stain_matrix = np.array([v2, v1])

        # Get concentrations
        concentrations = od_tissue @ np.linalg.pinv(stain_matrix).T
        max_concentrations = np.percentile(concentrations, 99, axis=0)

        return stain_matrix, max_concentrations

    def fit(self, reference_image: np.ndarray) -> 'StainNormalizer':
        """
        Fit normalizer to a reference image.

        Args:
            reference_image: RGB image to use as reference (H, W, 3), uint8

        Returns:
            self
        """
        image_float = reference_image.astype(np.float64) / 255.0
        image_float = np.clip(image_float, self.regularizer, 1.0)

        self.target_stain_matrix, self.target_concentrations = self._get_stain_matrix(
            image_float
        )
        self._target_stain_matrix_inv = np.linalg.pinv(self.target_stain_matrix)

        return self


class StainAugmentor:
    """
    Stain Augmentation for H&E images.

    Randomly perturbs the stain matrix and concentrations to create
    augmented versions of the image, simulating stain variation
    across different scanners and labs.

    Args:
        sigma_alpha: Standard deviation for stain matrix perturbation
        sigma_beta: Standard deviation for concentration perturbation
        luminosity_threshold: Threshold for background detection
    """

    def __init__(
        self,
        sigma_alpha: float = 0.2,
        sigma_beta: float = 0.2,
        luminosity_threshold: float = 0.8,
    ):
        self.sigma_alpha = sigma_alpha
        self.sigma_beta = sigma_beta
        self.luminosity_threshold = luminosity_threshold
        self.regularizer = 1e-6
        self._normalizer = StainNormalizer()

    def augment(
        self,
        image: np.ndarray,
        alpha: Optional[np.ndarray] = None,
        beta: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Apply stain augmentation to an image.

        Args:
            image: RGB image (H, W, 3), uint8 [0, 255]
            alpha: Stain matrix perturbation (2, 3). If None, randomly sampled.
            beta: Concentration perturbation (2,). If None, randomly sampled.

        Returns:
            Augmented RGB image (H, W, 3), uint8 [0, 255]
        """
        # Sample perturbations if not provided
        if alpha is None:
            alpha = np.random.normal(0, self.sigma_alpha, (2, 3))
        if beta is None:
            beta = np.random.normal(0, self.sigma_beta, 2)

        # Convert to float and optical density
        image_float = image.astype(np.float64) / 255.0
        image_float = np.clip(image_float, self.regularizer, 1.0)

        try:
            stain_matrix, _ = self._normalizer._get_stain_matrix(image_float)
        except Exception:
            # If stain estimation fails, apply simple color jittering
            return self._fallback_augment(image)

        # Perturb stain matrix
        perturbed_stain = stain_matrix * (1 + alpha)

        # Normalize rows
        perturbed_stain = perturbed_stain / (
            np.linalg.norm(perturbed_stain, axis=1, keepdims=True) + self.regularizer
        )

        # Get optical density
        od = -np.log(image_float + self.regularizer)
        od_flat = od.reshape(-1, 3)

        # Get concentrations with original stain matrix
        concentrations = od_flat @ np.linalg.pinv(stain_matrix).T

        # Perturb concentrations
        concentrations = concentrations * (1 + beta)

        # Reconstruct with perturbed stain matrix
        od_augmented = concentrations @ perturbed_stain
        od_augmented = od_augmented.reshape(od.shape)

        # Convert back to RGB
        image_augmented = np.exp(-od_augmented)
        image_augmented = np.clip(image_augmented * 255, 0, 255).astype(np.uint8)

        return image_augmented

    def _fallback_augment(self, image: np.ndarray) -> np.ndarray:
        """
        Fallback augmentation using simple color jittering.

        Args:
            image: RGB image (H, W, 3), uint8

        Returns:
            Augmented image
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)

        # Randomly adjust hue, saturation, value
        hsv[:, :, 0] = (hsv[:, :, 0] + np.random.uniform(-10, 10)) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * np.random.uniform(0.8, 1.2), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * np.random.uniform(0.8, 1.2), 0, 255)

        # Convert back to RGB
        image_augmented = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        return image_augmented

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply stain augmentation."""
        return self.augment(image)


if __name__ == "__main__":
    print("StainNormalizer and StainAugmentor modules loaded successfully.")

    # Test with synthetic image
    test_image = np.random.randint(100, 255, (256, 256, 3), dtype=np.uint8)

    normalizer = StainNormalizer()
    augmentor = StainAugmentor()

    # These will likely fail on random noise, but that's expected
    try:
        normalized = normalizer.normalize(test_image)
        print(f"Normalized image shape: {normalized.shape}")
    except Exception as e:
        print(f"Normalization skipped (expected for random noise): {e}")

    augmented = augmentor.augment(test_image)
    print(f"Augmented image shape: {augmented.shape}")
