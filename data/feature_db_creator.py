import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, "utils"))
)

from utils import preprocess, models
import argparse
import time
from pathlib import Path
from utils.logger import logger
from data import image_downloader
import pickle

IMG_DIR = image_downloader.IMG_DIR
FT_DIR = "./data/model_features"

def build_feature_database(csv_path, model_name):
    """Build feature database for valid images using a pretrained model"""
    feature_extractor = models.FeatureExtractor(model_name=model_name, use_cuda=False)
    feature_db = {}
    _, valid_images = preprocess.check_images_mapped_with_ids(IMG_DIR, csv_path)

    for image in valid_images:
        _, img_name = image
        image_path = os.path.join(IMG_DIR, img_name)
        image_tensor = preprocess.preprocess_image_from_file(image_path)
        feature_vector = feature_extractor.extract_features(image_tensor)
        feature_db[image] = feature_vector

    return feature_db


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", help="CSV file with image URLs")
    parser.add_argument(
        "--model",
        help="Name of the pretrained model to retrieve features. DEFAULT: 'resnet50'",
    )
    args = parser.parse_args()
    args.model = args.model or "resnet50"
    filepath = Path(
        os.path.join(FT_DIR, args.model, "feature_db.pkl")
    ) 

    t = time.perf_counter()
    feature_db = build_feature_database(args.csv_file, args.model)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Enregistrer la base de données des caractéristiques
    with open(filepath, "wb") as f:
        pickle.dump(feature_db, f)

    logger.info(
        f"Generated feature db for pretrained {args.model} in {time.perf_counter() - t:.2f} seconds"
    )
