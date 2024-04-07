import os
import glob
import argparse
import logging
from PIL import Image
import cv2
import pandas as pd

import utils
from configs import get_args_parse
from models import TransNetV2


class ImgAttrs:
    filepath: str
    vid_src: str
    frame_idx: int
    pts_time: float
    fps: float


class SplitKF:
    def __init__(self, args: dict, model_weight):
        self.args = args
        self.model = TransNetV2(model_dir=model_weight)
        self.fps: float = 25.0

    @utils.fcall
    def split(self, vid_dir: str, Ninterval: int, output_dir: str):
        vid_lst = glob.glob(vid_dir + "/*.mp4")
        vid_cnt = 0
        img_cnt = 0

        for vid_idx, vid_path in enumerate(vid_lst):
            vid_name = vid_path.split(".")[-2].split("/")[-1]
            (
                vid_frames,
                single_frame_pred,
                all_frame_pred,
            ) = self.model.predict_video(vid_path)
            scenes = self.model.predictions_to_scenes(all_frame_pred, threshold=0.1)

            logging.info(f"{len(scenes)} generated scenes")

            frame_idxs = list()
            cnt = 0

            img_lst = []
            for scene_idx, scene in enumerate(scenes):
                frame_idx = scene[0]
                intervalSize = (scene[-1] - scene[0]) / (Ninterval - 1)

                while frame_idx <= scene[-1]:
                    img_attrs = ImgAttrs()

                    # Image Attributes
                    img_attrs.vid_src = vid_path
                    img_attrs.frame_idx = int(frame_idx)
                    img_attrs.pts_time = frame_idx / self.fps
                    foldername = os.path.join(self.args.image_dir, vid_name)
                    img_attrs.filepath = os.path.join(
                        foldername,
                        "img_{}.jpg".format(cnt),
                    )

                    # Serialization
                    utils.ensure_path(foldername)
                    img_lst.append(img_attrs.__dict__)
                    frame_idxs.append((int(frame_idx), img_attrs.filepath))
                    frame_idx += intervalSize
                    cnt+=1
                    img_cnt += 1

            vid_cnt += 1

            # Serialization
            self.serialize_img_df(img_lst, output_dir, vid_name)
            self.serialize_img(vid_path, frame_idxs)


        logging.info("Total processed {} videos".format(vid_cnt))
        logging.info("Total processed {} images".format(img_cnt))

    def serialize_img_df(self, img_lst, output_dir, vid_name):
        pd.DataFrame(img_lst).to_csv(os.path.join(output_dir, "{}.csv".format(vid_name)))

    def serialize_img(self, vid_path, frame_idxs):

        cap = cv2.VideoCapture(vid_path)

        totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
        for frame_idx, image_path in frame_idxs:
            # check for valid frame number
            if frame_idx >= 0 & frame_idx <= totalFrames:
                # set frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES,frame_idx)
                
                ret, image = cap.read()

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                Image.fromarray(image).save(image_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Main Configuration", parents=[get_args_parse()])
    parser.add_argument(
        "--model-weight", type=str, default="./data/weights/transnetv2-weights/"
    )
    parser.add_argument("--enable-od", type=bool, default=False)
    args = parser.parse_args()

    # Setup
    utils.setup_logging()
    utils.ensure_path(args.image_dir)

    splitkf = SplitKF(args=args, model_weight=args.model_weight)
    splitkf.split(args.vid_dir, args.ninterval, args.image_dir)
