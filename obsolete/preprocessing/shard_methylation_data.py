"""
Methylation 데이터 샤딩 스크립트

396,065개의 probes를 10개 샤드로 분할하여 메모리 문제 해결
각 샤드는 variance가 높은 probes 위주로 구성
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from tqdm import tqdm

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_probe_variance(methylation_data):
    """
    각 probe의 variance 계산

    Args:
        methylation_data: DataFrame (patients × probes)

    Returns:
        variance_series: Series (probes, ) - 각 probe의 variance
    """
    logger.info("Calculating variance for all probes...")

    # 각 probe(column)의 variance 계산
    variances = methylation_data.var(axis=0)

    logger.info(f"Variance calculation complete")
    logger.info(f"  - Mean variance: {variances.mean():.6f}")
    logger.info(f"  - Median variance: {variances.median():.6f}")
    logger.info(f"  - Max variance: {variances.max():.6f}")
    logger.info(f"  - Min variance: {variances.min():.6f}")

    return variances


def create_shards(methylation_data, num_shards=10):
    """
    Methylation 데이터를 여러 샤드로 순차 분할

    ⚠️ 중요: Sequential split 사용
    - 모든 샤드를 사용하므로 variance 정렬 불필요
    - 추론 시 396K probes 전부 사용 (정보 손실 없음)

    Args:
        methylation_data: DataFrame (patients × probes)
        num_shards: 샤드 개수 (기본값: 10)

    Returns:
        shards: List of DataFrames
        shard_info: 샤드 메타데이터
    """
    total_probes = methylation_data.shape[1]
    total_patients = methylation_data.shape[0]

    logger.info(f"Creating {num_shards} shards from {total_probes:,} probes")
    logger.info(f"Total patients: {total_patients:,}")
    logger.info(f"Sharding method: Sequential (preserves original order)")

    # 순차적 분할 (variance 정렬 없음)
    sorted_probes = methylation_data.columns.tolist()

    # 샤드 생성
    shards = []
    shard_info = {
        'num_shards': num_shards,
        'total_probes': total_probes,
        'total_patients': total_patients,
        'method': 'sequential',
        'shards': []
    }

    probes_per_shard = total_probes // num_shards
    remainder = total_probes % num_shards

    start_idx = 0
    for shard_id in range(num_shards):
        # 마지막 샤드에 나머지 probes 추가
        end_idx = start_idx + probes_per_shard + (1 if shard_id < remainder else 0)

        # 현재 샤드의 probes
        shard_probes = sorted_probes[start_idx:end_idx]

        # DataFrame 생성
        shard_data = methylation_data[shard_probes].copy()
        shards.append(shard_data)

        # 메타데이터 저장
        shard_metadata = {
            'shard_id': shard_id,
            'num_probes': len(shard_probes),
            'num_patients': total_patients,
            'probe_range': f"{start_idx}-{end_idx}",
            'first_probe': shard_probes[0],
            'last_probe': shard_probes[-1],
            'probe_list': shard_probes  # 추론 시 필요
        }

        shard_info['shards'].append(shard_metadata)

        logger.info(f"Shard {shard_id}: {len(shard_probes):,} probes ({start_idx:,} - {end_idx:,})")

        start_idx = end_idx

    return shards, shard_info


def save_shards(shards, shard_info, output_dir, prefix='methylation_shard'):
    """
    샤드들을 parquet 파일로 저장

    Args:
        shards: List of DataFrames
        shard_info: 샤드 메타데이터
        output_dir: 출력 디렉토리
        prefix: 파일명 prefix
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving shards to {output_dir}")

    file_info = []

    for shard_id, shard_data in enumerate(tqdm(shards, desc="Saving shards")):
        # 파일명
        filename = f"{prefix}_{shard_id}.parquet"
        filepath = output_dir / filename

        # Parquet 저장
        shard_data.to_parquet(filepath, compression='snappy')

        # 파일 정보
        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        file_info.append({
            'shard_id': shard_id,
            'filename': filename,
            'size_mb': round(file_size_mb, 2),
            'shape': shard_data.shape
        })

        logger.info(f"  - {filename}: {shard_data.shape} ({file_size_mb:.2f} MB)")

    # 메타데이터에 파일 정보 추가
    shard_info['files'] = file_info
    shard_info['output_dir'] = str(output_dir)

    # 메타데이터 저장
    metadata_file = output_dir / f"{prefix}_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(shard_info, f, indent=2)

    logger.info(f"Metadata saved to {metadata_file}")

    # 요약 출력
    total_size_mb = sum([f['size_mb'] for f in file_info])
    logger.info(f"\n{'='*60}")
    logger.info(f"Sharding Summary:")
    logger.info(f"  - Total shards: {len(shards)}")
    logger.info(f"  - Total size: {total_size_mb:.2f} MB")
    logger.info(f"  - Average shard size: {total_size_mb/len(shards):.2f} MB")
    logger.info(f"  - Metadata file: {metadata_file}")
    logger.info(f"{'='*60}\n")


def verify_shards(output_dir, prefix='methylation_shard'):
    """
    저장된 샤드들 검증

    Args:
        output_dir: 샤드가 저장된 디렉토리
        prefix: 파일명 prefix

    Returns:
        verification_result: 검증 결과
    """
    output_dir = Path(output_dir)

    logger.info("Verifying saved shards...")

    # 메타데이터 로드
    metadata_file = output_dir / f"{prefix}_metadata.json"
    if not metadata_file.exists():
        logger.error(f"Metadata file not found: {metadata_file}")
        return False

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    # 각 샤드 파일 확인
    num_shards = metadata['num_shards']
    total_probes_found = 0

    for shard_id in range(num_shards):
        filename = f"{prefix}_{shard_id}.parquet"
        filepath = output_dir / filename

        if not filepath.exists():
            logger.error(f"Shard file not found: {filepath}")
            return False

        # 샤드 로드 (빠른 검증)
        shard_data = pd.read_parquet(filepath)
        total_probes_found += shard_data.shape[1]

        logger.info(f"✓ Shard {shard_id}: {shard_data.shape}")

    # 전체 probe 수 검증
    expected_probes = metadata['total_probes']

    if total_probes_found == expected_probes:
        logger.info(f"\n✅ Verification PASSED:")
        logger.info(f"  - All {num_shards} shards found")
        logger.info(f"  - Total probes: {total_probes_found:,} (expected: {expected_probes:,})")
        return True
    else:
        logger.error(f"\n❌ Verification FAILED:")
        logger.error(f"  - Probes found: {total_probes_found:,}")
        logger.error(f"  - Expected: {expected_probes:,}")
        return False


def main():
    """메인 실행 함수"""

    # 경로 설정
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'processed'
    input_file = data_dir / 'methylation_data_for_tabtransformer.parquet'
    output_dir = data_dir / 'methylation_shards'

    logger.info("="*60)
    logger.info("Methylation Data Sharding Script")
    logger.info("="*60)
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output directory: {output_dir}")

    # 데이터 로드
    logger.info("\nLoading methylation data...")
    logger.info("⚠️  This may take several minutes (29GB file)...")

    try:
        methylation_data = pd.read_parquet(input_file)
        logger.info(f"✓ Data loaded: {methylation_data.shape}")
        logger.info(f"  - Patients: {methylation_data.shape[0]:,}")
        logger.info(f"  - Probes: {methylation_data.shape[1]:,}")

    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # 샤드 생성 (순차적 분할)
    num_shards = 10
    shards, shard_info = create_shards(
        methylation_data,
        num_shards=num_shards
    )

    # 샤드 저장
    save_shards(shards, shard_info, output_dir)

    # 검증
    verify_shards(output_dir)

    logger.info("\n✅ Sharding complete!")
    logger.info(f"Sharded data saved to: {output_dir}")


if __name__ == '__main__':
    main()
