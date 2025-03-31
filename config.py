"""
config.py

Centralised configuration file for global variables, hyperparameters, and settings.
"""

# -----------------------------
# General Settings
# -----------------------------
PROJECT_ROOT = "/home/evammun/Thesis"  # Might be computed dynamically if needed.
SEED = 1
DEVICE = "cuda"  # Use "cuda" if available; otherwise, "cpu"

RETRAIN_MODELS = True

# -----------------------------
# Data Configuration
# -----------------------------
DATA_DIR = "/home/evammun/Thesis/Dataset"
DATA_PATH = f"{DATA_DIR}/MEDIC_train.tsv"

# Where newly created synthetic images (via Stable Diffusion) will be saved:
SYNTHETIC_DATA_DIR = "/home/evammun/Thesis/Dataset/data/synthetic"

# -----------------------------
# Training Hyperparameters
# -----------------------------
BATCH_SIZE = 32
NUM_THREADS = 5  # Adjust if necessary; standardise across modules
LEARNING_RATE = 1e-5
INITIAL_PATIENCE = 10

# -----------------------------
# (Legacy) Model Paths
# -----------------------------
MODEL_PATHS = {
    "resnet50": "models/best_model_resnet50.pth",
    "efficientnet_b1": "models/best_model_efficientnet_b1.pth",
    "mobilenet_v2": "models/best_model_mobilenet_v2.pth",
}

# -----------------------------
# Label Mappings
# -----------------------------
LABEL_MAPPINGS = {
    "damage_severity": {0: "little_or_none", 1: "mild", 2: "severe"},
    "informative": {0: "not_informative", 1: "informative"},
    "humanitarian": {
        0: "affected_injured_or_dead_people",
        1: "infrastructure_and_utility_damage",
        2: "not_humanitarian",
        3: "rescue_volunteering_or_donation_effort",
    },
    "disaster_types": {
        0: "earthquake",
        1: "fire",
        2: "flood",
        3: "hurricane",
        4: "landslide",
        5: "not_disaster",
        6: "other_disaster",
    },
}

# -----------------------------
# Number of Classes per Task
# -----------------------------
NUM_CLASSES = {
    "damage_severity": 3,
    "informative": 2,
    "humanitarian": 4,
    "disaster_types": 7,
}

# -----------------------------
# Callback and Other Training Settings
# -----------------------------
FINAL_PATIENCE = 5
VERBOSE = False
DELTA = 1e-7
LR_REDUCTION_FACTOR = 0.1

# -----------------------------
# Seaborn Theme Settings
# -----------------------------
SNS_STYLE = "white"
SNS_CONTEXT = "notebook"
SNS_FONT_SCALE = 1.1
SNS_PALETTE = "viridis_r"

# -----------------------------
# Task Names
# -----------------------------
TASKS = ["damage_severity", "informative", "humanitarian", "disaster_types"]

# -----------------------------
# API KEYS
# -----------------------------
OPENAI_KEY = ""
ANTHROPIC_KEY = ""
ANTHROPIC_MODEL = "claude-3-7-sonnet-20250219"
OPENAI_MODEL = "gpt-4o-2024-08-06"
MISTRAL_KEY = ""


# Stable Diffusion key and model:
STABLE_DIFFUSION_KEY = ""
STABLE_DIFFUSION_MODEL = "stable-diffusion-v1-6"

BLACK_FOREST_KEY = ""

# -----------------------------
# VISION MODELS
# -----------------------------
GPT_MODEL = "gpt-4o-2024-08-06"
SONNET_MODEL = "claude-3-7-sonnet-20250219"
HAIKU_MODEL = "claude-3-5-haiku-20241022"
PIXTRAL_LARGE_MODEL = "pixtral-large-2411"
PIXTRAL_SMALL_MODEL = "pixtral-12b-2409"

RELABELLING = False
RELABELLING_PROMPT = (
    RELABELING_PROMPT
) = """You are classifying an image according to four tasks.
Each task must have exactly one label from the provided list.

If multiple disasters appear, please choose the single most prominent one.
For damage severity, if a structure is partially damaged but still usable, label it 'mild'; if it is largely destroyed or unsafe to use, label it 'severe'.

TASK 1: "damage_severity"
 - little_or_none: no visible disaster damage or extremely minimal.
 - mild: partial structural damage (e.g. up to ~50% damage, partly collapsed roof).
 - severe: significant destruction (structure no longer usable or mostly collapsed).

TASK 2: "informative"
 - not_informative: not helpful for disaster relief (random images, ads, blurred, etc.).
 - informative: clearly shows disaster impact, damage, or relief efforts.

TASK 3: "humanitarian"
 - affected_injured_or_dead_people: the image shows people who are physically harmed, displaced, or casualties.
 - infrastructure_and_utility_damage: visible damage to buildings, roads, power lines, etc.
 - not_humanitarian: does not show anything relevant for disaster relief.
 - rescue_volunteering_or_donation_effort: depicts active rescue, donation, or volunteering.

TASK 4: "disaster_types"
 - earthquake: collapsed buildings/roads typical of seismic activity.
 - fire: noticeable flames, smoke, or burnt structures.
 - flood: submerged roads/buildings, high water level.
 - hurricane: storm surge, strong wind damage, fallen power lines/trees.
 - landslide: fallen earth/rock, mudslides, collapsed ground.
 - not_disaster: the image does not show any disaster.
 - other_disaster: any other catastrophe (explosion, vehicle crash, war, etc.).

**Return only a JSON object** with exactly these four keys:
{
  "damage_severity": "...",
  "informative": "...",
  "humanitarian": "...",
  "disaster_types": "..."
}
using only the allowed label strings above. No additional text or explanation."""

SYNTHETIC_EVALUATION_PROMPT = (
    SYNTHETIC_EVALUATION_PROMPT
) = """As a Humanitarian Crisis Image Analyst, examine this image and provide your assessment. Some images may show actual disasters while others may show normal scenes with no disaster present. First, analyze what you see, then classify it according to standard humanitarian response categories.

<analysis>
If this is not a photo from a disaster scene, but a news article, infographic, etc...be sure to specify this is a non-disaster image.
Examine the image carefully and describe what you see related to potential disaster, damage, humanitarian concerns, and informativeness. If no disaster is present, simply describe what you see in the image. Don't mention specific label categories yet, just describe what's visually present.
</analysis>

Now, based on your analysis, classify this image according to the following categories. If more than one disaster is depicted (for example, a hurricane causing flooding or fire), pick the original cause (e.g. hurricane).
1. DAMAGE SEVERITY:
   - little_or_none: non-disaster image or image with damage-free infrastructure (except for normal wear and tear)
   - severe: substantial destruction making infrastructure non-livable/non-usable
   - mild: partially destroyed buildings/bridges/houses/roads (approximately up to 50% damage)
   

2. INFORMATIVENESS:
   - informative: useful for humanitarian aid or disaster response
   - not_informative: not useful for humanitarian aid (newspaper article, ads, logos, cartoons, blurred images, minor incident)

3. HUMANITARIAN CATEGORY:
   - affected_injured_or_dead_people: shows people affected by the disaster- homeless, trapped, injured, etc...
   - infrastructure_and_utility_damage: shows built structures (buildings, roads...) affected/damaged by a disaster (not decay, poverty, etc...)
   - not_humanitarian: non-disaster image or not relevant for humanitarian aid
   - rescue_volunteering_or_donation_effort: shows rescue, volunteering, or response efforts, including first responders, emergency vehicles, donations, etc..

4. DISASTER TYPE:
   - earthquake
   - fire: man-made fires,  wildfires, vehicles on fire, etc...
   - flood: flooded areas, houses, roads, other infrastructures, only if NOT not caused by hurricane
   - hurricane: include flooding from hurricanes, as well as wind damage, etc...
   - landslide: landslide, mudslide, landslip, rockfall
   - not_disaster: anything not easily linked to any disaster type
   - other_disaster: plane crash, bus/car/train accident, explosion, war, conflicts, etc... or where the cause of the disaster is unclear

<labels>
{
  "damage_severity": "",
  "informative": "",
  "humanitarian": "",
  "disaster_types": ""
}
</labels>"""

# -----------------------------
# Miscellaneous Settings
# -----------------------------
CONF_SAVE_DIR = "results/confusion_matrices"
CROSS_TASK_SAVE_DIR = "results/cross_task_errors"

# ==========================================================
# New Dictionary-Based Dataset Versioning & Model Paths
# ==========================================================
# Possible values: "original", "relabelled", or "augmented"
DATASET_VERSION = "augmented"

DATA_VERSIONS = {
    "original": {
        "train": "/home/evammun/Thesis/Dataset/MEDIC_train.tsv",
        "val": "/home/evammun/Thesis/Dataset/MEDIC_dev.tsv",
        "test": "/home/evammun/Thesis/Dataset/MEDIC_test.tsv",
    },
    "relabelled": {
        "train": "/home/evammun/Thesis/Dataset/MEDIC_train_relabelled.tsv",
        "val": "/home/evammun/Thesis/Dataset/MEDIC_dev_relabelled.tsv",
        "test": "/home/evammun/Thesis/Dataset/MEDIC_test_relabelled.tsv",
    },
    # New "augmented" version reuses relabelled dev/test splits, but references MEDIC_train_augmented.tsv for training
    "augmented": {
        "train": "/home/evammun/Thesis/Dataset/MEDIC_train_augmented.tsv",
        "val": "/home/evammun/Thesis/Dataset/MEDIC_dev_relabelled.tsv",
        "test": "/home/evammun/Thesis/Dataset/MEDIC_test_relabelled.tsv",
    },
}


def get_data_paths():
    """
    Return a dictionary with 'train', 'val', 'test' keys
    pointing to the correct TSV files for the current DATASET_VERSION.
    """
    return DATA_VERSIONS[DATASET_VERSION]


# Models paths referencing subfolders
MODEL_PATHS_VERSIONS = {
    "original": {
        "resnet50": "models/original/best_model_resnet50.pth",
        "efficientnet_b1": "models/original/best_model_efficientnet_b1.pth",
        "mobilenet_v2": "models/original/best_model_mobilenet_v2.pth",
    },
    "relabelled": {
        "resnet50": "models/relabelled/best_model_resnet50.pth",
        "efficientnet_b1": "models/relabelled/best_model_efficientnet_b1.pth",
        "mobilenet_v2": "models/relabelled/best_model_mobilenet_v2.pth",
    },
    # We can store future augmented-model checkpoints here if needed:
    "augmented": {
        "resnet50": "models/augmented/best_model_resnet50.pth",
        "efficientnet_b1": "models/augmented/best_model_efficientnet_b1.pth",
        "mobilenet_v2": "models/augmented/best_model_mobilenet_v2.pth",
    },
}


def get_model_paths():
    """
    Return a dictionary mapping each model name
    to its checkpoint path for the current DATASET_VERSION.
    """
    return MODEL_PATHS_VERSIONS.get(DATASET_VERSION, {})


MODEL_DIRS = {
    "original": "models/original",
    "relabelled": "models/relabelled",
    "augmented": "models/augmented",
}


def get_model_dir():
    """
    Return the top-level folder where trained model checkpoints should be saved,
    depending on DATASET_VERSION.
    """
    return MODEL_DIRS.get(DATASET_VERSION, "models")


# Path to the CSV file of prompt templates for LLM-based augmentation
AUGMENTATION_PROMPTS_FILE = (
    "/home/evammun/Thesis/WSLcode/src/augmentation/llm_prompts.json"
)

SYNTHETIC_IMAGE_GENERATION = (
    "/home/evammun/Thesis/WSLcode/src/augmentation/test_captions_fallback.csv"
)
