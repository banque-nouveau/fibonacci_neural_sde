# Global configuration for the amgm package.

from pathlib import Path

""" The directory where the amgm package is located. """
project_root = Path(__file__).resolve().parents[1]

""" Workspace directory, for temporary files. """
ws_dir = project_root / "workspace"

""" Dataset root directory """
dataset_root = project_root / "Data"

""" AM dataset directory """
am_dataset_dir = dataset_root / "data-20250505"

""" RX1 dataset directory """
rx1_dataset_dir = dataset_root / "FICC" / "SEBx_rates_bund_vs_macro_indicators.csv"

ws_dir.mkdir(parents=True, exist_ok=True)


def work_dir(name: str, create: bool = True) -> Path:
    """ Get a directory to store temporary or intermediate files in.
    Args:
        name (str): Name of the directory to create.
        create (bool): Whether to create the directory if it does not exist.
    Returns: Path to workspace / name
    """
    path = ws_dir / name
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path