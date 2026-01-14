"""
Example Usage of WSI Preprocessing Pipeline
============================================

This script demonstrates how to use the preprocessing pipeline
for different scenarios.
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)


def example_single_wsi():
    """Process a single WSI file."""
    from wsi_model.src.preprocessing import WSIPreprocessor
    from wsi_model.src.preprocessing.wsi_preprocessor import PreprocessingConfig

    # Configure preprocessing
    config = PreprocessingConfig(
        patch_size=256,              # 256x256 patches
        min_tissue_ratio=0.8,        # At least 80% tissue in patch
        stain_normalize=True,        # Apply Macenko stain normalization
        model_name='swin_tiny',      # Use Swin-T for feature extraction
        batch_size=64,               # Batch size for GPU inference
    )

    # Create preprocessor
    preprocessor = WSIPreprocessor(
        output_dir='./output/features',
        config=config,
        device='cuda:0',  # Use GPU
    )

    # Process single WSI
    result = preprocessor.process_wsi(
        slide_path='path/to/slide.svs',
        slide_id='patient_001',  # Optional custom ID
    )

    # Check result
    if result.success:
        print(f"Successfully processed: {result.slide_id}")
        print(f"  Valid patches: {result.num_patches_valid}")
        print(f"  Features saved to: {result.features_path}")
    else:
        print(f"Failed: {result.error_message}")


def example_batch_processing():
    """Process all WSI files in a directory."""
    from wsi_model.src.preprocessing import WSIPreprocessor
    from wsi_model.src.preprocessing.wsi_preprocessor import PreprocessingConfig

    config = PreprocessingConfig(
        patch_size=256,
        stain_normalize=True,
        model_name='swin_tiny',
        batch_size=64,
    )

    preprocessor = WSIPreprocessor(
        output_dir='./output/features',
        config=config,
    )

    # Process all SVS files in directory
    results = preprocessor.process_directory(
        input_dir='path/to/wsi_folder',
        pattern='*.svs',
        recursive=True,  # Search subdirectories
    )

    # Get summary
    summary = preprocessor.get_summary(results)
    print(f"\nProcessing Summary:")
    print(f"  Total slides: {summary['total_slides']}")
    print(f"  Successful: {summary['successful']}")
    print(f"  Failed: {summary['failed']}")
    print(f"  Total patches: {summary['total_patches']}")


def example_tissue_detection_only():
    """Only perform tissue detection without feature extraction."""
    from wsi_model.src.preprocessing import TissueDetector, PatchExtractor
    import numpy as np

    # Open slide
    extractor = PatchExtractor(patch_size=256)
    slide = extractor.open_slide('path/to/slide.svs')

    # Get thumbnail
    thumbnail, scale_factor = extractor.get_thumbnail(slide, size=2048)

    # Detect tissue
    detector = TissueDetector()
    result = detector.detect(thumbnail, return_contours=True)

    print(f"Tissue ratio: {result.tissue_ratio:.1%}")
    print(f"Number of tissue regions: {len(result.contours)}")

    # Get patch coordinates
    coords = detector.get_tissue_coordinates(
        mask=result.mask,
        patch_size=256,
        min_tissue_ratio=0.8,
        scale_factor=scale_factor,
    )

    print(f"Valid patch locations: {len(coords)}")

    # Visualize (optional)
    vis = detector.visualize_detection(thumbnail, result.mask)
    # Save or display vis...

    slide.close()


def example_stain_augmentation():
    """Apply stain augmentation to patches."""
    from wsi_model.src.preprocessing import StainNormalizer
    from wsi_model.src.preprocessing.stain_normalizer import StainAugmentor
    import numpy as np

    # Load a sample patch (256x256x3, uint8)
    patch = np.random.randint(100, 255, (256, 256, 3), dtype=np.uint8)

    # Normalize to reference stain
    normalizer = StainNormalizer()
    normalized = normalizer.normalize(patch)

    # Apply random augmentation
    augmentor = StainAugmentor(
        sigma_alpha=0.2,  # Stain matrix perturbation
        sigma_beta=0.2,   # Concentration perturbation
    )

    augmented = augmentor.augment(patch)
    print(f"Original shape: {patch.shape}")
    print(f"Augmented shape: {augmented.shape}")


def example_feature_extraction():
    """Extract features from patches without full pipeline."""
    from wsi_model.src.preprocessing import FeatureExtractor
    from wsi_model.src.preprocessing.feature_extractor import FeatureStorage
    import numpy as np

    # Create feature extractor
    extractor = FeatureExtractor(
        model_name='swin_tiny',  # Options: resnet50, swin_tiny, swin_base, etc.
        device='cuda:0',
        batch_size=64,
    )

    print(f"Feature dimension: {extractor.feature_dim}")

    # Extract features from list of patches
    patches = [np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(100)]
    coordinates = np.random.randint(0, 10000, (100, 2))

    features, coords = extractor.extract_features(patches, coordinates)
    print(f"Features shape: {features.shape}")  # (100, 768)

    # Save features
    FeatureStorage.save_single_slide(
        output_dir='./output/features',
        slide_id='test_slide',
        features=features,
        coordinates=coords,
        model_name='swin_tiny',
    )

    # Load features
    loaded_features, loaded_coords, metadata = FeatureStorage.load_features(
        input_path='./output/features/test_slide.h5',
        slide_id='test_slide',
    )
    print(f"Loaded features: {loaded_features.shape}")


def example_tcga_preprocessing():
    """
    Complete example for preprocessing TCGA WSI data.

    Expected directory structure:
        wsi_model/data/raw/
        ├── BLCA/
        │   ├── TCGA-XX-XXXX/
        │   │   └── TCGA-XX-XXXX-01A-01-TS1.xxx.svs
        │   └── ...
        ├── BRCA/
        │   └── ...
        └── ...
    """
    from wsi_model.src.preprocessing import WSIPreprocessor
    from wsi_model.src.preprocessing.wsi_preprocessor import PreprocessingConfig
    import os

    # Paths
    raw_dir = os.path.join(project_root, 'wsi_model/data/raw')
    processed_dir = os.path.join(project_root, 'wsi_model/data/processed')

    # Configuration matching Cancer Cell paper approach
    config = PreprocessingConfig(
        # Patch settings
        patch_size=256,              # 256x256 patches (can also use 512)
        patch_level=0,               # Full resolution (20x typically)
        min_tissue_ratio=0.8,        # 80% tissue threshold
        blur_threshold=50.0,         # Laplacian variance threshold

        # Stain normalization
        stain_normalize=True,        # Macenko normalization
        stain_augment=False,         # Augmentation during training, not preprocessing

        # Feature extraction
        model_name='swin_tiny',      # Swin-T (768-dim features)
        batch_size=64,               # Adjust based on GPU memory
        num_workers=4,               # Data loading workers

        # Output
        save_patches=False,          # Don't save patch images (saves disk space)
        save_features=True,          # Save feature vectors
    )

    # Create preprocessor
    preprocessor = WSIPreprocessor(
        output_dir=processed_dir,
        config=config,
        device='cuda:0',
    )

    # Process all cancer types
    cancer_types = ['BLCA', 'ACC', 'BRCA']  # Add more as downloaded

    for cancer in cancer_types:
        cancer_dir = os.path.join(raw_dir, cancer)
        if not os.path.exists(cancer_dir):
            print(f"Skipping {cancer}: directory not found")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {cancer}")
        print(f"{'='*60}")

        results = preprocessor.process_directory(
            input_dir=cancer_dir,
            pattern='*.svs',
            recursive=True,
        )

        summary = preprocessor.get_summary(results)
        print(f"\n{cancer} Summary:")
        print(f"  Processed: {summary['total_slides']}")
        print(f"  Successful: {summary['successful']}")
        print(f"  Total patches: {summary['total_patches']}")


if __name__ == "__main__":
    print("WSI Preprocessing Pipeline - Example Usage")
    print("=" * 60)
    print("\nAvailable examples:")
    print("  1. example_single_wsi() - Process single WSI")
    print("  2. example_batch_processing() - Process directory")
    print("  3. example_tissue_detection_only() - Tissue detection only")
    print("  4. example_stain_augmentation() - Stain augmentation")
    print("  5. example_feature_extraction() - Feature extraction only")
    print("  6. example_tcga_preprocessing() - Full TCGA preprocessing")
    print("\nRun the desired example by uncommenting in main block.")

    # Uncomment to run specific example:
    # example_single_wsi()
    # example_batch_processing()
    # example_tissue_detection_only()
    # example_stain_augmentation()
    # example_feature_extraction()
    # example_tcga_preprocessing()
