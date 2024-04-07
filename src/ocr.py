import os
import cv2
import pandas as pd
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from paddleocr import PaddleOCR
from PIL import Image
from typing import Any
from configs import get_args_parse


class OCR:
    def __init__(self) -> None:
        self.text_detector = PaddleOCR(use_angle_cls=True, lang="en")
        config = Cfg.load_config_from_name("vgg_seq2seq")
        config["cnn"]["pretrained"] = True
        config["device"] = "cuda:0"
        config["predictor"]["beamsearch"] = False
        self.text_recognitor = Predictor(config)

        parser = get_args_parse()
        args = parser.parse_args()
        self.df_directory = args.ocr_dir

    def text_detection(self, image_path: str) -> Any:
        """input: path of image
        output: list of bounding boxes
        """
        return self.text_detector.ocr(image_path)

    def save_text_boxes(self, image_path: str) -> list:
        """input: path of image
        output: list of file path where the bounding boxes stored
        """

        detector_result = self.text_detection(image_path)
        bouding_boxes = [
            detector_result[0][i][0] for i in range(len(detector_result[0]))
        ]

        image = cv2.imread(image_path)
        count = 0

        splitted_directory = image_path.split("/")
        directory = "/".join(splitted_directory.remove(splitted_directory[-1]))
        save_directory = os.path.join(directory, image_path.split(".")[0])

        for box in bouding_boxes:
            y_min, y_max, *other = [int(point[0]) for point in box]
            x_min, *other, x_max = [int(point[1]) for point in box]
            region = image[x_min:x_max, y_min:y_max]
            cv2.imwrite(f"{save_directory}/output_{count}.jpg", region)

        file_path = [
            f"{save_directory}/output_{index}" for index in range(0, count + 1)
        ]
        return file_path

    def text_recognition(self, image_path: str) -> tuple:
        """input: image path
        output: tuple of image path and text
        """
        image_path_list = []
        text_list = []
        file_path_list = self.text_detection(image_path)
        for path in file_path_list:
            image = Image.open(path)
            text = self.text_recognitor.predict(image)
            image_path_list.append(image_path)
            text_list.append(text)
        return image_path_list, text_list

    def __call__(self, directory: str):
        """
        input: directory store image
        output: None
        process OCR through all image in this directory
        then save result into a csv file
        """
        result_dict = {"n": [], "text": []}
        for path in os.listdir(directory):
            image_path = os.path.join(directory, path)
            image_path_list, text_list = self.text_recognition(image_path)
            for image_path, text in zip(image_path_list, text_list):
                result_dict["n"].append(path)
                result_dict["text"].append(text)

        result_df = pd.DataFrame(result_dict)
        result_df.to_csv(f"{self.df_directory}/{directory}.csv", index=False)


if __name__ == "__main__":
    ocr_tools = OCR()
    ocr_tools(directory="data/keyframes/L01_V001")
