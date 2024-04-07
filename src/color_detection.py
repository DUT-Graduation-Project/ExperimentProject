import json
import os
import cv2
import numpy as np
import argparse
import pandas as pd
import shutil
from colorthief import ColorThief


class ColorDetection:
    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self.rgb_dict = {
            "red": [255, 0, 0],
            "green": [0, 255, 0],
            "blue": [0, 0, 255],
            "yellow": [255, 255, 0],
            "pink": [255, 192, 203],
            "white": [255, 255, 255],
            "black": [0, 0, 0],
            "grey": [128, 128, 128],
            "brown": [150, 75, 0],
            "purple": [128, 0, 128],
        }

    def _load_json_content(self, file_path: str):
        content = open(file_path)
        return json.load(content)

    def _denormalize_box(self, box, image_shape):
        """Scales corner box coordinates from normalized values to image dimensions.
        # Arguments
            box: Numpy array containing corner box coordinates.
            image_shape: List of integers with (height, width).
        # Returns
            returns: box corner coordinates in image dimensions
        """
        # x_min, y_min, x_max, y_max = box[:4]
        y_min, x_min, y_max, x_max = box[:4]

        height, width = image_shape
        x_min = int(float(x_min) * width)
        y_min = int(float(y_min) * height)
        x_max = int(float(x_max) * width)
        y_max = int(float(y_max) * height)

        return [y_min, x_min, y_max, x_max]

    def _read_image(self, image_path: str):
        return cv2.imread(image_path)

    def _filter_by_threshold(self, file_path: str):
        json_content = self._load_json_content(file_path)
        scores = []
        boxes = []
        entities = []
        for score, box, entity in zip(
            json_content["detection_scores"],
            json_content["detection_boxes"],
            json_content["detection_class_entities"],
        ):
            if float(score) >= self.threshold:
                scores.append(float(score))
                boxes.append(box)
                entities.append(entity)
        return scores, boxes, entities

    def _rgb_to_cielab(self, rgb_values: list):
        rgb_values = [i / 255 for i in rgb_values]
        rgb_array = np.array(rgb_values, dtype=np.float32).reshape(1, 1, 3)
        cielab_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2Lab)
        cielab_values = cielab_array[0, 0, :]
        return cielab_values

    def _euclidean_distance(self, dominant_color: list, base_color: list) -> float:
        return np.sum(
            np.square(np.power((np.array(dominant_color) - np.array(base_color)), 2))
        )

    def _human_perception_euclidean(self, dominant_color: list, base_color: list):
        return np.sum(
            np.array([0.3, 0.59, 0.11])
            * np.power((np.array(dominant_color) - np.array(base_color)), 2)
        )

    def _cielab_euclidean(self, dominant_color: list, base_color: list):
        return np.sum(
            np.square(
                np.power(
                    (
                        np.array(self._rgb_to_cielab(dominant_color))
                        - np.array(self._rgb_to_cielab(base_color))
                    ),
                    2,
                )
            )
        )

    def _ranking(self, distances: dict):
        return {k: v for k, v in sorted(distances.items(), key=lambda item: item[1])}

    def _color_detect(self, file_path: str, image_path: str):
        result_dict = {"file_path": [], "object": [], "color": []}
        _, boxes, entities = self._filter_by_threshold(file_path)
        image = self._read_image(image_path)
        count = 0
        os.makedirs("image_test", exist_ok=True)
        for box, entity in zip(boxes, entities):
            y_min, x_min, y_max, x_max = self._denormalize_box(box, image.shape[:2])
            object_image = image[y_min:y_max, x_min:x_max]

            cv2.imwrite(f"image_test/test_{count}.jpg", object_image)

            color_thief = ColorThief(f"image_test/test_{count}.jpg")
            dominant_color = color_thief.get_color(quality=1)
            distances = self._ranking(
                {
                    color: self._euclidean_distance(
                        dominant_color, self.rgb_dict[color]
                    )
                    for color in self.rgb_dict.keys()
                }
            )
            result_dict["file_path"].append(file_path.split(".")[0])
            result_dict["object"].append(entity)
            result_dict["color"].append(list(distances.keys())[0])
            count += 1
        shutil.rmtree("image_test")
        return pd.DataFrame(result_dict)

    def __call__(self, file_directory: str, directory: str):
        final_df = pd.DataFrame()
        for image in os.listdir(directory):
            image_path = os.path.join(directory, image)
            file_path = os.path.join(file_directory, f"{image.split('.')[0]}.json")
            final_df = pd.concat(
                [final_df, self._color_detect(file_path, image_path)], axis=0
            )

        return final_df


if __name__ == "__main__":
    detector = ColorDetection(threshold=0.4)
    final_df = detector(
        file_directory="L001_V001", directory="keyframe/keyframe-1/L001_V001"
    )
    print(final_df)
