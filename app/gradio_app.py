import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, "utils"))
)

import gradio as gr
from PIL import Image, UnidentifiedImageError
from utils import preprocess, models
from utils.logger import logger
from data import image_downloader

IMG_DIR = image_downloader.IMG_DIR


def predict_gradio(image_file, model_name, nb_similar_images, ignore_duplicate):
    try:
        image_tensor = preprocess.preprocess_image_from_file(image_file.name)
        logger.info(f"Processed image: {image_file.name}")

        result = models.predict(
            image_tensor=image_tensor,
            image_name=os.path.basename(image_file.name),
            model_name=model_name,
            ignore_duplicate=ignore_duplicate,
            nb_similar_images=nb_similar_images,
        )
        logger.info(
            f"Successfully retrieved top {nb_similar_images} similar images: {result}"
        )
        logger.info(
            f"Successfully processed the image and retrieved top {nb_similar_images} similar images."
        )
        similar_images_with_labels = [
            (
                Image.open(os.path.join(IMG_DIR, img["name"])),
                f"Score de similitude: {img['score']:2.2%}",
            )
            for img in result
        ]
        return Image.open(image_file.name), similar_images_with_labels

    except UnidentifiedImageError:
        logger.error("The uploaded file is not a valid image.")
        return "Error: The uploaded file is not a valid image.", []
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return f"Error: File not found: {e}", []
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return f"Error: {str(e)}", []


with gr.Blocks() as iface:
    # Input section
    with gr.Row(equal_height=False):
        image_input = gr.File(label="Image File")
        model_name_input = gr.Dropdown(
            ["resnet50", "vgg16"], value="resnet50", label="Model Name"
        )
        nb_similar_images_input = gr.Slider(
            1, 10, step=1, value=3, label="Number of Similar Images"
        )
        ignore_duplicate_input = gr.Checkbox(label="Ignore Duplicate", value=True)

    # Output section
    with gr.Row():
        with gr.Column(scale=1):
            input_image_output = gr.Image(type="pil", label="Input Image")
        with gr.Column(scale=3):
            similar_images_output = gr.Gallery(
                label="Similar Images",
                type="pil",
                preview=True,
                show_label=True,
            )

    # Define the prediction function
    predict_button = gr.Button("Find closest images")
    predict_button.click(
        fn=predict_gradio,
        inputs=[
            image_input,
            model_name_input,
            nb_similar_images_input,
            ignore_duplicate_input,
        ],
        outputs=[input_image_output, similar_images_output],
    )

if __name__ == "__main__":
    iface.launch()
