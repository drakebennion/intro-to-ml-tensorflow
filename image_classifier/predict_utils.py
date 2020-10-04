from PIL import Image

import argparse
import json
import numpy as np
import tensorflow as tf

def get_args():
    parser = argparse.ArgumentParser(description='Path to image, tensorflow model, top k, category names')

    parser.add_argument('image_path')
    parser.add_argument('model_path')
    # require top_k to be 1 <= k <= len(...something ha)
    parser.add_argument('--top_k', action='store', default=1, type=int)
    parser.add_argument('--category_names', action='store')

    return parser.parse_args()

def process_image(image):
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, [224, 224])
    image /= 255
    return image.numpy()

def process_image_for_prediction(image):
    # prepare image to be fed into our predictor
    image = Image.open(image)
    image_np = np.asarray(image)
    processed_image = process_image(image_np)
    return np.expand_dims(processed_image, axis=0)

def process_labels(labels, label_map):
    if label_map:
        with open(label_map, 'r') as f:
            class_names = json.load(f)
        labels = [class_names[label] for label in labels]
    return labels