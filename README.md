# Synthetic Data Augmentation with Stable Diffusion for the MEDIC Dataset

This repository demonstrates how to augment a multi-task image classification dataset ([MEDIC](https://crisisnlp.qcri.org/medic/)) using synthetic images created with generative AI models (diffusion models), with the ultimate goal of improving CNN performance under class imbalance. Due to ethical constraints, we **do not include** the original disaster images or the generated synthetic images in this repository. Please see the note below on how to obtain them.

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Running the Experiments](#running-the-experiments)
4. [Datasets and Ethical Constraints](#datasets-and-ethical-constraints)
5. [Dependencies](#dependencies)
6. [Contact](#contact)

---

## Overview

This codebase explores the use of **Stable Diffusion** (and related diffusion models) to generate synthetic images for the [MEDIC dataset](https://paperswithcode.com/dataset/medic) (Alam et al., 2023). The primary aim is to address class imbalance in disaster‐related imagery by supplementing underrepresented categories with newly generated images. We follow a systematic pipeline:

- Experiment 1:
  - **Zero-Shot Classification**
- Experiment 2:
  1. **Relabelling**
  2. **Synthetic Image Generation** (using LLMs to generate captions, then passing them to Stable Diffusion)
  3. **Augmented Training** with a mix of real and synthetic images

<details>
  <summary>Notes</summary>

- In the project report, we first presented the synthetic data augmentation (Experiment 2) and then our results with zero-shot classification (here Experiment 1).
- Experiment 1 was conducted on the relabelled dataset.

</details>

---

## Repository Structure

Below is a high-level view of the repository. The key code for running experiments resides in the `experiments/` directory, which contains two main subfolders:

```
experiments/
├─ experiment1
│ ├─ 01_zero_shot_classification.ipynb
│ └─ utils.py
└─ experiment2/
├─ 01_original_training.ipynb
├─ 02_relabelling_pipeline.ipynb
├─ 03_train_relabelled.ipynb
├─ 04_synthetic_image_generation.ipynb
└─ 05_train_augmented.ipynb home/ models/ results/ tensorboard/
```

- **`experiment1/`**

  - `01_zero_shot_classification.ipynb`: Demonstrates zero-shot classification performance.
  - `utils.py`: Shared utility code for setting up notebooks (logging, seeds, GPU detection, etc.).

- **`experiment2/`**

  - `01_original_training.ipynb`: Trains a baseline model on the original dataset.
  - `02_relabelling_pipeline.ipynb`: Applies the relabelling step to the dataset.
  - `03_train_relabelled.ipynb`: Retrains the model on the newly relabelled data.
  - `04_synthetic_image_generation.ipynb`: Uses large language models (LLMs) to generate captions and then Stable Diffusion or other models to create synthetic images.
  - `05_train_augmented.ipynb`: Combines real and synthetic data for training, examining the impact on class‐imbalanced tasks.

- **Other Folders**
  - `home/`, `models/`, `results/`, `tensorboard/`: Organisational folders for logs, trained model checkpoints, intermediate results, and TensorBoard logs.

---

## Running the Experiments

1. **Clone the repository** (or download the source code).
2. **Install dependencies** (see the [Dependencies](#dependencies) section).
3. **Set up configuration** (e.g., API keys, dataset paths) by editing `config.py` (referenced in the code).
   - Be careful not to accidentally expose your API keys. For example, do not upload the modified `config.py` to a public repository.
5. **Open the desired Jupyter notebooks** in the `experiments/` subfolders.
6. **Run each notebook** cell-by-cell, following the instructions at the top of each notebook.

> **Note**: The recommended order for the main pipeline is:
>
> 1. `experiment2/01_original_training.ipynb`
> 2. `experiment2/02_relabelling_pipeline.ipynb`
> 3. `experiment2/03_train_relabelled.ipynb`
> 4. `experiment2/04_synthetic_image_generation.ipynb`
> 5. `experiment2/05_train_augmented.ipynb`

You can also explore `experiment1/01_zero_shot_classification.ipynb` independently to see alternative baselines.

---

## Datasets and Ethical Constraints

- **Original MEDIC dataset**: We do **not** include the original disaster images here. They are freely available from the [MEDIC listing on Papers with Code](https://paperswithcode.com/dataset/medic).
- **Synthetic dataset**: Our synthetic images are also **not** included in this repository due to ethical considerations and restrictions on sharing disaster imagery. If you wish to obtain them, please contact the author and provide:
  1. A clear statement of your **ethical use case**.
  2. **Written confirmation** that the images will not be further redistributed.

Because of the sensitive nature of disaster imagery, we strongly encourage responsible use of these materials and compliance with local IRB or ethics board guidelines.

---

## Dependencies

Below is a concise list of Python libraries required to run the experiments end‐to‐end. Please ensure your environment is properly set up with the following:

- **Python Standard Library**:  
  `os, sys, pathlib, tempfile, random, math, logging, warnings, inspect, base64, collections, typing, csv, re, time, datetime, threading, concurrent.futures, mimetypes, json, multiprocessing`
- **Third‐Party Libraries**:
  - `numpy`
  - `torch` (PyTorch)
  - `nvidia-dali`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `scipy`
  - `IPython`
  - `tueplots`
  - `umap-learn` (optional, for UMAP-based embedding visualisation)
  - `requests`
  - `tqdm`
- **LLM and Diffusion Model Integration**:
  - `openai` (for GPT-based calls)
  - `anthropic` (for Claude-based calls)
  - _(Optionally)_ `transformers`, `diffusers`, `accelerate` (if running local Stable Diffusion or alternative pipelines)

A typical installation might look like:

```bash
pip install numpy torch nvidia-dali pandas matplotlib seaborn scikit-learn scipy ipython tueplots umap-learn requests tqdm openai anthropic
```

---

### License

Code in this project is released under the terms of the [MIT License](https://github.com/evammun/genai-data-aug-disasters/blob/main/LICENSE).

### Contact

If you have any questions or wish to obtain the synthetic images, please reach out via email at `evammun 📧 gmail.com`.
In the latter case, make sure to provide details about your intended use, and be prepared to sign an agreement not to redistribute the images.

* * * * *

**Thank you for your interest in Synthetic Data Augmentation for the MEDIC dataset.** 
We hope you find these experiments interesting useful for exploring class imbalance solutions in disaster‐related contexts.
