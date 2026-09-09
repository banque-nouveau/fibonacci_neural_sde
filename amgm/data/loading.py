import pickle
from pathlib import Path
from subprocess import run
from typing import Union
from warnings import warn

import pandas as pd
from tqdm import tqdm

from amgm import config


def download(src_url: str, dst_path: Union[Path, str]):
    """Download a directory of files from Google Cloud Storage.
    Args:
        src_url (str): The source URL in the format "gs://bucket_name/path/to/files".
        dst_path (Union[Path, str]): The destination path where files will be downloaded.
    """
    Path(dst_path).mkdir(parents=True, exist_ok=True)
    run(["gcloud", "storage", "cp", "--no-clobber", "-r", src_url, str(dst_path)], check=True)


def download_sebx_am_data(dset_name: str = "data-20250505"):
    """Download the asset-management dataset from the bucket named "sebx-asset-management".
    The dataset will be stored in the directory amgm.config.dataset_root / dset_name
    Args:
        dset_name (str): Name of the dataset to download, e.g. "data-20250505".
    """
    print(f"Downloading {dset_name}...")
    data_dir = config.dataset_root / dset_name
    download(f"gs://sebx-asset-management/{dset_name}/", data_dir)


def load_sebx_am_data(data_dir: Path, cached=True, force=False):
    """Load all .csv (and .txt) data files in the AM dataset from the specified directory
    into a dictionary of DataFrames.
    Some columns are cast to specific types, e.g. "IssueId" to category and "Date" to datetime.
    The data is cached in a file named "cache.pkl" in the same directory.
    Args:
        data_dir (Path): Directory containing the data files.
        cached (bool): Whether to use cached data, which loads faster.
        force (bool): Force-reload the cached data even if it exists.
    """

    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} does not exist. Please download the dataset first.")

    data = dict()
    data_cache = data_dir / "cache.pkl"

    if not cached or not data_cache.exists() or force:
        # Load, then cache
        for file in tqdm(sorted(data_dir.glob("*"))):
            if file.suffix.lower() not in (".csv", ".txt"):
                continue

            # Cast specific columns to types
            df = pd.read_csv(file, delimiter="\t", dtype={"IssueId": str, "VoAdj": float}, parse_dates=["Date"])

            data[file.stem] = df

        if cached and not data_cache.exists() or force:
            with open(data_cache, "wb") as f:
                pickle.dump(data, f)

    else:
        # Load from cache
        with open(data_cache, "rb") as f:
            data = pickle.load(f)

    return data
