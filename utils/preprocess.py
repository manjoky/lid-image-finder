import os
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
import pandas as pd
from utils.logger import logger


def check_images_mapped_with_ids(image_dir: str, csv_path: str):
    """
    Check images in a directory against IDs from a CSV file and identify valid and invalid images.

    Parameters:
        image_dir (str): Directory containing the images to check.
        csv_path (str): Path to the CSV file with image names and their corresponding IDs.

    Returns:
        tuple:
            - A list of tuples with ID and name for invalid images.
            - A list of tuples with ID and name for valid images.
    """
    invalid_images = []
    valid_images = []
    images_df = pd.read_csv(csv_path)
    images_df["name"] = [el[-1] for el in images_df["url"].str.split("/")]
    for img_file in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_file)
        id = images_df[images_df["name"].eq(img_file)].id.values[0]
        try:
            with Image.open(img_path) as img:
                valid_images.append((id, img_file))
                img.verify()
        except (UnidentifiedImageError, IOError) as e:
            invalid_images.append((id, img_file))
            logger.warn(f"Corrupted image {img_file} found: {e}")

    logger.info(f"Number of valid images: {len(valid_images)}")
    return invalid_images, valid_images


def preprocess_image_from_file(image_path: str, target_size: tuple = (224, 224)):
    """
    Preprocess the image by resizing, normalising and converting it to a tensor.

    Parameters:
        image_path (str): The path to the image file.
        target_size (tuple): The target size for resizing the image (default 224x224).

    Returns:
        torch.Tensor: The tensor of the pre-processed image.
    """
    preprocess_pipeline = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    try:
        image = Image.open(image_path).convert("RGB")

        # Run the preprocessing pipeline
        preprocessed_image = preprocess_pipeline(image)
        return preprocessed_image

    except UnidentifiedImageError:
        logger.error(
            f"Error: the '{image_path}' image cannot be identified and will be ignored."
        )
        return None
