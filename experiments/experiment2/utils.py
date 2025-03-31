"""
utils.py

Common setup utilities for experiment notebooks.
"""

import sys
import os
import tempfile
import random
import logging
import warnings
from pathlib import Path

import numpy as np
import torch
import seaborn as sns

# For IPython magic commands:
from IPython import get_ipython


def setup_notebook():
    """
    1. Add the top-level project directory (WSLcode/) to sys.path.
    2. Load autoreload.
    3. Suppress unwanted warnings/logging.
    4. Redirect stderr to a temp file.
    5. Clear CUDA cache.
    6. Import config after adjusting sys.path.
    7. Set random seeds, device, and seaborn style according to config.
    """

    # 1) Ensure we add WSLcode/ to sys.path (two levels up from this file)
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    # 2) Automatically reload all modules
    ip = get_ipython()
    if ip is not None:
        ip.run_line_magic("load_ext", "autoreload")
        ip.run_line_magic("autoreload", "2")

    # 3) Suppress warnings and set DALI logging level
    warnings.filterwarnings("ignore", category=UserWarning, module="libpng")
    warnings.filterwarnings("ignore", category=UserWarning, module="nvidia.dali")
    logging.getLogger("nvidia.dali").setLevel(logging.ERROR)

    # 4) Redirect stderr to a temporary file
    temp = tempfile.NamedTemporaryFile(delete=False)
    os.dup2(temp.fileno(), sys.stderr.fileno())

    # 5) Clear CUDA cache
    torch.cuda.empty_cache()

    # 6) Now that project_root is in sys.path, import config
    import config

    # 7) Set random seeds, device, and seaborn style
    seed = config.SEED
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[setup_notebook] Using device: {device}")

    sns.set(style=config.SNS_STYLE)
