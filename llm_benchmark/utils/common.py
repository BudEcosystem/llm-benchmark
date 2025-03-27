import hashlib
import json
import random
from pathlib import Path
import pyarrow.parquet as pq
from typing import Optional

from datasets import load_dataset
from modelscope.msdatasets import MsDataset

#get hash of a string or object
def get_hash(obj):
    return hashlib.sha256(str(obj).encode()).hexdigest()

def get_dataset_files(datasets, extensions=None, is_folder=False):
    """
    Convert a list of dataset folders into a list of file paths.
    
    :param datasets: List of dataset folder paths.
    :param extensions: Set of allowed file extensions (e.g., {".json", ".jsonl", ".parquet"}).
    :return: List of file paths grouped by dataset.
    """
    all_files = []
    
    for dataset_folder in datasets:
        dataset_path = Path(dataset_folder)
        if is_folder:
            if not dataset_path.is_dir():
                raise ValueError(f"Invalid dataset directory: {dataset_folder}")
            
            # Fetch all files within the directory (recursively)
            dataset_files = [str(file) for file in dataset_path.rglob("*") if file.is_file()]
        else:
            if not dataset_path.is_file():
                raise ValueError(f"Invalid dataset file: {dataset_folder}")
            
            # Fetch the file itself
            dataset_files = [str(dataset_path)]
            
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


def distribute_samples_evenly(datasets, concurrency):
    num_datasets = len(datasets)
    base_samples_per_dataset = concurrency // num_datasets
    remainder = concurrency % num_datasets  # Remaining samples to distribute

    dataset_samples_mapping = {}

    for i, dataset in enumerate(datasets):
        extra_sample = 1 if i < remainder else 0  # Distribute remainder evenly
        dataset_samples_mapping[dataset.id] = base_samples_per_dataset + extra_sample

    return dataset_samples_mapping


def handle_hf_datasets(dataset, seed, dataset_sample_size=None):
    random.seed(seed)
    data = load_dataset(dataset.hf_hub_url, split=dataset.split or "train")

    data_samples = get_formatted_samples_from_dataset(data, dataset, dataset_sample_size, seed)
    return data_samples

def handle_modelscope_datasets(dataset, seed, dataset_sample_size=None):
    random.seed(seed)
    data = MsDataset.load(dataset.ms_hub_url, split=dataset.split or 'train')

    data_samples = get_formatted_samples_from_dataset(data, dataset, dataset_sample_size, seed)
    return data_samples
    

def handle_local_datasets(dataset, seed, is_folder=False, dataset_sample_size=None):
    random.seed(seed)
    dataset_files = get_dataset_files([dataset.folder], extensions=[".jsonl", ".parquet", ".json"], is_folder=is_folder)[0]
    
    dataset_samples = []
    random.shuffle(dataset_files)
    for dataset_path in dataset_files:
        with open(dataset_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                data_samples = get_formatted_samples_from_dataset(data, dataset, dataset_sample_size, seed)
                dataset_samples.extend(data_samples)
                if len(dataset_samples) < dataset_sample_size:
                    dataset_sample_size = dataset_sample_size - len(dataset_samples)
                    continue
                return dataset_samples
            else:
                raise TypeError("Invalid data format: expected list of dicts.")
                

def get_formatted_samples_from_dataset(data: list, dataset, dataset_sample_size: int, seed: int):
    random.seed(seed)
    random.shuffle(data)
    
    default_column_mapping = {"prompt": "instruction", "query": "input", "response": "output", "chosen": None, "messages": "conversations"}
    dataset_column_mapping = dataset.columns or {}
    default_tags_mapping = {"role_tag": "from", "content_tag": "value", "user_tag": "human", "assistant_tag": "gpt"}
    dataset_tags_mapping = dataset.tags or {}
    columns = columns = {k: dataset_column_mapping[k] if k in dataset_column_mapping else v 
           for k, v in default_column_mapping.items()}
    tags = {k: dataset_tags_mapping[k] if k in dataset_tags_mapping else v 
           for k, v in default_tags_mapping.items()}
    formatting = dataset.formatting or 'alpaca'

    samples = []
    for sample in data:
        formatted_sample = {"prompt": "", "response": ""}
        if formatting == "alpaca":
            formatted_sample["prompt"] = f"{sample.get(columns['prompt'], '')}\n{sample.get(columns['query'], '')}"
            formatted_sample["response"] = sample.get(columns['chosen'], '') if columns.get('chosen') else sample.get(columns['response'], '')
        else:
            messages = sample.get(columns['messages'], '')
            if len(messages) > 2:
                for msg in messages:
                    if msg.get(tags["role_tag"], "") == tags["user_tag"]:
                        formatted_sample["prompt"] = f"{msg.get(tags['content_tag'], '')}"
                    elif msg.get(tags["role_tag"], "") == tags["assistant_tag"]:
                        formatted_sample["response"] = f"{msg.get(tags['content_tag'], '')}"
        if formatted_sample["prompt"] and formatted_sample["response"]:
            samples.append(formatted_sample)
            if len(samples) >= dataset_sample_size:
                break

    return samples
            

def combine_multiple_datasets(
    concurrency: int,
    seed: int,
    datasets: Optional[list] = None,
    ) -> Optional[dict]:
    """
    Efficiently sample prompts from multiple datasets without loading entire datasets into memory.
    
    # :param datasets: List of dataset paths (each dataset is a list of files).
    :param datasets: List of dataset objects.
    :param concurrency: Total number of prompts to select.
    :return: List of sampled prompts.
    """
    if not datasets:
        return None

    random.seed(seed)

    dataset_samples_mapping = distribute_samples_evenly(datasets, concurrency)

    combined_data = {}
    
    for dataset in datasets:
        if dataset.hf_hub_url:
            combined_data[dataset.id] = handle_hf_datasets(dataset, seed, dataset_sample_size=dataset_samples_mapping[dataset.id])
        elif dataset.ms_hub_url:
            combined_data[dataset.id] = handle_modelscope_datasets(dataset, seed, dataset_sample_size=dataset_samples_mapping[dataset.id])
        elif dataset.script_url:
            raise NotImplementedError("Loading dataset using script url is currently not supported.")
        elif dataset.file_name:
            combined_data[dataset.id] = handle_local_datasets(dataset, seed, dataset_sample_size=dataset_samples_mapping[dataset.id])
        elif dataset.folder:
            try:
                combined_data[dataset.id] = handle_local_datasets(dataset, seed, is_folder=True, dataset_sample_size=dataset_samples_mapping[dataset.id])
            except ValueError:
                raise NotImplementedError
        else:
            raise ValueError("Invalid dataset type.")
        
    return combined_data