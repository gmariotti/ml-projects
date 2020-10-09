import click
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import pprint
import json


def process_image(image: str) -> np.ndarray:
    image = Image.open(image)
    image = np.asarray(image)
    image = tf.convert_to_tensor(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image /= 255
    image = np.expand_dims(image.numpy(), axis=0)
    return image


def prediction(image_to_predict: np.ndarray, model_path: str, top_k: int) -> [np.ndarray, np.ndarray]:
    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})
    result = model.predict(image_to_predict)
    # top_k if present
    top_k = top_k if top_k != 0 else len(result[0])
    probs, indexes = tf.math.top_k(result[0], k=top_k)
    return probs.numpy(), (indexes.numpy() + 1).astype(str)


def map_to_names(category_names: str, classes: np.ndarray):
    with open(category_names, 'r') as f:
        class_names = json.load(f)
        result = [class_names[index] for index in classes]
        return result


@click.command()
@click.argument("image", type=click.Path(exists=True, file_okay=True, readable=True))
@click.argument("model", type=click.Path(exists=True, file_okay=True, readable=True))
@click.option(
    "--top_k",
    default=0,
    type=click.IntRange(0, None),
    help="Return the top K most likely classes. A value of 0 means all probabilities will be displayed."
)
@click.option(
    "--category_names",
    type=click.Path(exists=True, file_okay=True, readable=True),
    help="Path to a JSON file mapping labels to flower names"
)
def predict(image: str, model: str, top_k: int, category_names: str):
    # load image
    image_to_predict = process_image(image)
    # run prediction
    probs, classes = prediction(image_to_predict, model, top_k)
    # map to category_names if present
    if category_names is not None:
        classes = map_to_names(category_names, classes)
    # print results
    probs = np.vectorize(lambda p: np.format_float_positional(p, precision=4))(probs)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(dict(zip(classes, probs)))


if __name__ == '__main__':
    predict()
