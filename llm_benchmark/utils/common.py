import hashlib
import json
import random
from pathlib import Path
import pyarrow.parquet as pq
from typing import Optional

#get hash of a string or object
def get_hash(obj):
    return hashlib.sha256(str(obj).encode()).hexdigest()

def get_dataset_files(datasets, extensions=None):
    """
    Convert a list of dataset folders into a list of file paths.
    
    :param datasets: List of dataset folder paths.
    :param extensions: Set of allowed file extensions (e.g., {".json", ".jsonl", ".parquet"}).
    :return: List of file paths grouped by dataset.
    """
    all_files = []
    
    for dataset_folder in datasets:
        dataset_path = Path(dataset_folder)
        if not dataset_path.is_dir():
            raise ValueError(f"Invalid dataset directory: {dataset_folder}")
        
        # Fetch all files within the directory (recursively)
        dataset_files = [str(file) for file in dataset_path.rglob("*") if file.is_file()]
        
        # Filter by allowed extensions (if provided)
        if extensions:
            dataset_files = [file for file in dataset_files if Path(file).suffix in extensions]
        
        all_files.append(dataset_files)
    
    return all_files


def stream_jsonl(file_path, sample_size):
    """Stream JSONL file and perform reservoir sampling."""
    sampled = []
    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            if len(sampled) < sample_size:
                sampled.append(data)
            else:
                j = random.randint(0, i)
                if j < sample_size:
                    sampled[j] = data
    return sampled


def stream_parquet(file_path, sample_size):
    """Stream Parquet file in chunks and perform reservoir sampling."""
    sampled = []
    pf = pq.ParquetFile(file_path)
    row_count = pf.metadata.num_rows  # Total rows across all row groups
    num_row_groups = pf.num_row_groups

    step = max(row_count // sample_size, 1)  # Step size to avoid full load

    for i in range(sample_size):
        row_idx = random.randint(i * step, min((i + 1) * step - 1, row_count - 1))

        # Find the correct row group
        row_group_idx = 0
        accumulated_rows = 0
        for j in range(num_row_groups):
            rows_in_group = pf.metadata.row_group(j).num_rows
            if accumulated_rows + rows_in_group > row_idx:
                row_group_idx = j
                break
            accumulated_rows += rows_in_group

        # Read the row group and get the specific row
        row_group = pf.read_row_group(row_group_idx).to_pandas()
        row_within_group_idx = row_idx - accumulated_rows
        sampled.append(row_group.iloc[row_within_group_idx].to_dict())

    return sampled


def stream_json(file_path, sample_size):
    """Read JSON file and randomly sample."""
    with open(file_path, "r") as f:
        data = json.load(f)
    return random.sample(data, min(sample_size, len(data)))


def sample_from_dataset(dataset_files, sample_size):
    """Sample prompts from multiple files in a dataset."""
    sampled_prompts = []
    
    for file_path in dataset_files:
        ext = Path(file_path).suffix.lower()
        if ext == ".jsonl":
            sampled_prompts.extend(stream_jsonl(file_path, sample_size))
        elif ext == ".parquet":
            sampled_prompts.extend(stream_parquet(file_path, sample_size))
        elif ext == ".json":
            sampled_prompts.extend(stream_json(file_path, sample_size))
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        # If we have enough samples, stop
        if len(sampled_prompts) >= sample_size:
            return sampled_prompts[:sample_size]
    
    return sampled_prompts


def combine_multiple_datasets(datasets, concurrency) -> Optional[list]:
    """
    Efficiently sample prompts from multiple datasets without loading entire datasets into memory.
    
    :param datasets: List of dataset paths (each dataset is a list of files).
    :param concurrency: Total number of prompts to select.
    :return: List of sampled prompts.
    """
    if not datasets:
        return None
    num_datasets = len(datasets)
    base_samples_per_dataset = concurrency // num_datasets
    remainder = concurrency % num_datasets  # Remaining prompts to distribute
    
    datasets_with_files = get_dataset_files(datasets, extensions=[".jsonl", ".parquet", ".json"])

    combined_data = []

    for i, dataset_files in enumerate(datasets_with_files):
        samples_needed = base_samples_per_dataset + (1 if i < remainder else 0)
        combined_data.extend(sample_from_dataset(dataset_files, samples_needed))

    return combined_data