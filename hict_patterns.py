import string
import tensorflow as tf
import cooler
import numpy as np
import os
from matplotlib import pyplot as plt
import matplotlib.colors as clr
import io

def __predict_matrix_center(matrix,model, device_name, resolution):
    clr_map_center = clr.LinearSegmentedColormap.from_list('yarg', ['#000', '#fff'], N=256)
    max_value = np.nanmax(matrix)
    min_value = np.nanmin(matrix)

    detections = []
    for i in range(10, matrix.shape[0] - 10):
        point = (i, i)
        point_area = matrix[point[1] - 10:point[1] + 10, point[0] - 10:point[0] + 10]
        buf = io.BytesIO()
        plt.imsave(buf, point_area,
                   cmap=clr_map_center, vmax=max_value, vmin=min_value)

        with tf.device(device_name):
            img = tf.keras.utils.load_img(
                buf, target_size=(20, 20)
            )
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)

            score = tf.nn.softmax(np.array(model(img_array))[0])

            if np.argmax(score) == 1:
                detections.append(i * resolution)
    return detections


def __predict_matrix_final(matrix, model, center_detections, device_name, resolution):
    clr_map = clr.LinearSegmentedColormap.from_list('yarg', ['#e6e6e6', '#000'], N=256)
    max_value = np.nanmax(matrix)
    min_value = np.nanmin(matrix)
    detections = []
    for x in center_detections:
        for y in center_detections:
            if x <= y:
                continue
            point = (x // resolution, y // resolution)
            point_area = matrix[point[1] - 10:point[1] + 10, point[0] - 10:point[0] + 10]
            buf = io.BytesIO()
            plt.imsave(buf, point_area, cmap=clr_map,
                       vmax=max_value, vmin=min_value)

            with tf.device(device_name):
                img = tf.keras.utils.load_img(
                    buf, target_size=(20, 20)
                )
                img_array = tf.keras.utils.img_to_array(img)
                img_array = tf.expand_dims(img_array, 0)

                score = tf.nn.softmax(np.array(model(img_array))[0])

                if np.argmax(score) == 1:
                    detections.append((x, y))
    return detections


def __predict_cooler_single_chromosome(resolution: int, chromname: string, device_name, model_centered, model_final, cooler_file):
    matrix = np.log10(cooler_file.matrix(balance=False).fetch(chromname) + 1)
    center_detections = __predict_matrix_center(matrix, model_centered, device_name, resolution)
    detected_points = __predict_matrix_final(matrix, model_final, center_detections, device_name, resolution)
    return detected_points


def predict_cooler_single_chromosome(cooler_file_path: string, resolution: int, chromname: string, device_name: string = '/device:CPU:0'):
    model_centered = tf.keras.models.load_model("artifacts/center_model.keras")
    model_final = tf.keras.models.load_model("artifacts/final_model.keras")
    c = cooler.Cooler(f'{cooler_file_path}::/resolutions/{resolution}')
    matrix = np.log10(c.matrix(balance=False).fetch(chromname) + 1)
    center_detections = __predict_matrix_center(matrix, model_centered, device_name, resolution)
    detected_points = __predict_matrix_final(matrix, model_final, center_detections, device_name, resolution)
    return detected_points


def predict_cooler(cooler_file_path: string, resolution:int, interchromosome: bool = True, device_name: string = '/device:CPU:0'):
    c = cooler.Cooler(f'{cooler_file_path}::/resolutions/{resolution}')
    model_centered = tf.keras.models.load_model("artifacts/center_model.keras")
    model_final = tf.keras.models.load_model("artifacts/final_model.keras")
    if interchromosome:
        result = {}
        for chr_name in c.chromnames:
            result[chr_name] = __predict_cooler_single_chromosome(cooler_file_path, resolution, chr_name, device_name,
                                                                  model_centered, model_final)
        return result
    else:
        matrix = np.log10(c.matrix(balance=False)[0:] + 1)
        center_detections = __predict_matrix_center(matrix, model_centered, device_name, resolution)
        detected_points = __predict_matrix_final(matrix, model_final, center_detections, device_name, resolution)
        return detected_points
