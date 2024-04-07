import os
import argparse
import pandas as pd
import json
from tqdm import tqdm
import tensorflow_hub as hub
import tensorflow as tf

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

import utils
from configs import get_args_parse

# Tensorflow v1
os.environ["TFHUB_MODEL_LOAD_FORMAT"] = "UNCOMPRESSED"


class ObjectDetection:
    def __init__(self, args: dict):
        self.args = args
        # self.initialize_sess()

    def identify_detector(self):
        model_name = self.args.od_model
        if "tfhub" in model_name:
            detector = hub.Module(model_name)
            return detector

    def load_img(self, path: str):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        return img

    def run_detector(self, detector, path):
        img = self.load_img(path)

        converted_img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
        result = detector(converted_img)

        result = {key: value.numpy() for key, value in result.items()}
        return result

    @utils.fcall
    def detect_with_directory(self, img_dir: str, batch_size: int = 50):
        # Image DataFrame
        img_df = pd.read_csv(os.path.join(img_dir, "img_df.csv"))
        detector = self.identify_detector()
        # Object Detection Process
        for idx, row in tqdm(img_df.iterrows()):
            imgpath = row["filename"]
            vidname, imgname = imgpath.split("/")[-2], imgpath.split("/")[-1]
            imgname = imgname.split(".")[-2]

            with tf.gfile.Open(imgpath, "rb") as binfile:
                image_string = binfile.read()

            detector_output, _ = self.run_detector(detector=detector, path=image_string)

            # Ensure output dir
            output_dir = os.path.join(self.args.object_dir, vidname)
            utils.ensure_path(output_dir)

            # Serialization
            self.serialize(detector_output, output_dir, imgname + ".json")

    def __call__(self, imgpath):
        with tf.gfile.Open(imgpath, "rb") as binfile:
            image_string = binfile.read()

        detector = self.identify_detector()
        detector_output, _ = self.run_detector(detector=detector, path=image_string)

        return detector_output

    def serialize(self, detector_output: dict, output_dir: str, filename: str):
        with open(os.path.join(output_dir, filename), "w") as outfile:
            json_str = json.dumps(detector_output, cls=utils.NumpyEncoder)
            outfile.write(json_str)
            outfile.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Main Configuration", parents=[get_args_parse()])

    args = parser.parse_args()

    utils.setup_logging()

    utils.ensure_path(args.object_dir)

    model = ObjectDetection(args)

    model.detect_with_directory(args.image_dir)
