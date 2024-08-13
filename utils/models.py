import os
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from pathlib import Path
from data import image_downloader, feature_db_creator

IMG_DIR = image_downloader.IMG_DIR
FT_DIR = feature_db_creator.FT_DIR


class FeatureExtractor:
    def __init__(self, model_name: str = "resnet50", use_cuda: bool = False):
        """
        Initialize the FeatureExtractor with a pre-trained model.

        Parameters:
            model_name (str): The name of the pre-trained model to use.
            use_cuda (bool): Flag to determine whether to use GPU (not used for this project)
        """
        self.device = torch.device(
            "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        )

        # Load a pre-trained model
        if model_name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1
            self.model = models.resnet50(weights=weights)
            self.model = nn.Sequential(
                *list(self.model.children())[:-1]
            )  # Remove the last fully connected layer
        elif model_name == "vgg16":
            weights = models.VGG16_Weights.IMAGENET1K_V1
            self.model = models.vgg16(
                weights=weights
            ).features  # Use only the feature extractor part
        else:
            raise ValueError(f"Model {model_name} is not supported.")

        self.model = self.model.to(self.device)
        self.model.eval()

    def extract_features(self, image_tensor: torch.Tensor):
        """
        Extract feature vector from an image tensor using the pre-trained model.

        Parameters:
            image_tensor (torch.Tensor): The preprocessed image tensor.

        Returns:
            np.ndarray: The extracted feature vector.
        """
        image_tensor = image_tensor.unsqueeze(0).to(
            self.device
        )  # Add batch dimension and move to the correct device

        with torch.no_grad():  # Disable gradient calculation
            features = self.model(image_tensor)

        features = features.cpu().numpy().flatten()

        return features


def compute_similarity(
    feature_vector_1: np.ndarray, feature_vector_2: np.ndarray, metric: str = "cosine"
):
    """
    Measure the similarity between two feature vectors.

    Parameters:
        feature_vector_1 (np.ndarray): The first feature vector.
        feature_vector_2 (np.ndarray): The second feature vector.
        metric (str): The similarity metric to use ('cosine' or 'euclidean').

    Returns:
        float: The similarity score between the two vectors.
    """
    if metric == "cosine":
        similarity = cosine_similarity([feature_vector_1], [feature_vector_2])[0][0]
    elif metric == "euclidean":
        similarity = -np.linalg.norm(feature_vector_1 - feature_vector_2)
    else:
        raise ValueError(f"Metric {metric} is not supported.")

    return float(similarity)


def predict(
    image_tensor: torch.Tensor,
    image_name: str,
    model_name: str,
    nb_similar_images: int,
    ignore_duplicate: bool,
):
    """
    Find and return the most similar images to the given image based on feature similarity.

    Parameters:
        image_tensor (torch.Tensor): The tensor representation of the input image.
        image_name (str): The name of the input image.
        model_name (str): The name of the model used for feature extraction (e.g., "resnet50", "vgg16").
        nb_similar_images (int): The number of similar images to return.
        ignore_duplicate (bool): Whether to ignore the input image if it's already in the database.

    Returns:
        list: A list of the top similar images with their ID, name, and similarity score.
    """
    filepath = Path(os.path.join(FT_DIR, model_name, "feature_db.pkl"))
    with open(filepath, "rb") as f:
        feature_db = pickle.load(f)

    feature_extractor = FeatureExtractor(model_name=model_name)
    feature_db = (
        {k: v for k, v in feature_db.items() if k[1] != image_name}
        if ignore_duplicate
        else feature_db
    )
    query_features = feature_extractor.extract_features(image_tensor)

    similarities = []
    for keys, features in feature_db.items():
        id, name = keys
        similarity = compute_similarity(query_features, features)
        similarities.append(
            dict(zip(("id", "name", "score"), (int(id), name, similarity)))
        )

    similarities.sort(key=lambda x: x["score"], reverse=True)
    top_nb = similarities[:nb_similar_images]

    return top_nb
