"""
model_setup.py

Defines functions for setting up the model architecture for multi-task classification.
A pre-trained base model is adapted by replacing its final classifier with a custom multi-task head.
Default values (such as the number of classes per task) are obtained from the central configuration.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ResNet50_Weights,
    EfficientNet_B1_Weights,
    MobileNet_V2_Weights,
)

# Import default number of classes from the central config.
from config import NUM_CLASSES


class CustomModel(nn.Module):
    """
    A custom neural network model that adapts a pre-trained base model for multi-task classification.

    Parameters
    ----------
    base_model : torch.nn.Module
        A pre-trained model from which features are extracted.
    num_classes : dict
        Dictionary specifying the number of classes for each task.
    model_name : str
        Name of the base model, used to determine architecture-specific modifications.

    The class creates task-specific classification heads and applies a flattening layer before them.
    """

    def __init__(self, base_model, num_classes, model_name):
        """
        Initialize the CustomModel by setting up the feature extractor and classification heads.
        """
        super(CustomModel, self).__init__()
        # Set up feature extraction layers based on model name.
        if "resnet" in model_name:
            self.features = nn.Sequential(
                *list(base_model.children())[
                    :-1
                ],  # Exclude the final fully connected layer
                nn.AdaptiveAvgPool2d((1, 1)),  # Add Adaptive Average Pooling
            )
            in_features = base_model.fc.in_features

        elif "mobilenet_v2" in model_name:
            self.features = nn.Sequential(
                *list(base_model.children())[:-1],  # Exclude the classifier layer
                nn.AdaptiveAvgPool2d(
                    (1, 1)
                ),  # Ensure Adaptive Average Pooling is included
            )
            in_features = 1280  # Fixed for MobileNet V2

        elif "efficientnet" in model_name:
            self.features = nn.Sequential(
                *list(base_model.children())[:-1],  # Exclude the final classifier layer
                nn.AdaptiveAvgPool2d((1, 1)),  # Explicitly define pooling
            )
            in_features = base_model.classifier[1].in_features

        else:
            raise ValueError(f"Unsupported model name: {model_name}")

        # Flatten the output features.
        self.flatten = nn.Flatten()

        # Create a dictionary of linear layers (classification heads) for each task.
        self.class_heads = nn.ModuleDict(
            {
                "disaster_types": nn.Linear(in_features, num_classes["disaster_types"]),
                "informative": nn.Linear(in_features, num_classes["informative"]),
                "humanitarian": nn.Linear(in_features, num_classes["humanitarian"]),
                "damage_severity": nn.Linear(
                    in_features, num_classes["damage_severity"]
                ),
            }
        )

    def forward(self, x):
        """
        Forward pass: extract features, flatten them, and apply each classification head.

        Parameters
        ----------
        x : Tensor
            Input data tensor.

        Returns
        -------
        dict
            Dictionary mapping each task to its raw output logits.
        """
        x = self.features(x)
        x = self.flatten(x)
        outputs = {task: head(x) for task, head in self.class_heads.items()}
        return outputs


def initialize_model(model_name, num_classes=None, feature_extract=False):
    """
    Initialize a pre-trained model with a custom head for multi-task classification.

    Args
    ----
    model_name : str
        Name of the model to initialize ('resnet50', 'efficientnet_b1', 'mobilenet_v2').
    num_classes : dict, optional
        A dictionary containing the number of classes for each task.
        Defaults to NUM_CLASSES from the central configuration.
    feature_extract : bool, optional
        If True, freeze the base model parameters.

    Returns
    -------
    torch.nn.Module
        A PyTorch model ready for training.
    """
    # Use centralized defaults if num_classes is not provided.
    if num_classes is None:
        from config import NUM_CLASSES

        num_classes = NUM_CLASSES

    if model_name == "resnet50":
        base_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    elif model_name == "efficientnet_b1":
        base_model = models.efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)
    elif model_name == "mobilenet_v2":
        base_model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    else:
        raise ValueError("Invalid model name")

    # Freeze base model parameters if using feature extraction.
    if feature_extract:
        for param in base_model.parameters():
            param.requires_grad = False

    model = CustomModel(base_model, num_classes, model_name)
    return model


def extract_features(self, x):
    """
    Extract features using the base model. (This function may be used if the full forward pass is not desired.)
    """
    return self.base_model(x)
