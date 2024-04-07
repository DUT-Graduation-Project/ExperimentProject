import os
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np

from utils import load_data

class FrameAttr:
    # Dir
    n : int
    framename : str
    filepath : str
    vid_src : str
    
    # Attrs
    pts_time : float
    fps : float
    frame_idx : float

    # External
    objects : list
    caption : str
    texts : list
    clip_feat : np.array
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type = str, default = "./data/")
    args = parser.parse_args()
    
    OCR_THRESHOLD = 0.8
    OD_THRESHOLD = 0.6
    frame_attrs = list()

    for root, dirs, frames in tqdm(os.walk(os.path.join(args.path,"keyframe"), topdown = False)):
        if len(root.split("/")) == 5:
            vid_src = root.split("/")[-1]
            key_num = root.split("/")[-2]

            map_keyframes = pd.read_csv(os.path.join(args.path, "map-keyframes", vid_src) + ".csv", index_col=0)
            OCR_features = pd.read_csv(os.path.join(args.path, "ocr", key_num, vid_src) + ".csv")
            caption_features = pd.read_csv(os.path.join(args.path, "captions", key_num, vid_src) + ".csv")


            for frame in frames:
                frame_attr = FrameAttr()
                
                # Attrs
                frame_attr.n = int(frame.split(".")[-2])
                frame_attr.framename = frame
                frame_attr.filepath = os.path.join(root, frame)
                frame_attr.vid_src = vid_src

                # Map_Keyframes
                attr = map_keyframes.loc[frame_attr.n]
                frame_attr.pts_time = attr["pts_time"]
                frame_attr.fps = attr["fps"]
                frame_attr.frame_idx = attr["frame_idx"]

                # Objects
                objects = load_data(os.path.join(args.path, "objects", vid_src, frame.split(".")[-2]) + ".json")
                objects = [objects["detection_class_entities"][idx] for idx in range(len(objects["detection_class_entities"])) if float(objects["detection_scores"][idx]) > OD_THRESHOLD]
                objects = list(set(objects))
                objects = [str(object).lower() for object in objects]
                frame_attr.objects = objects

                # OCR Text
                ocr = OCR_features[OCR_features["n"] == frame]
                ocr_text_lst = ocr[ocr["score"] > OCR_THRESHOLD]["text"].to_list()
                ocr_text_lst = [str(ocr_text).lower() for ocr_text in ocr_text_lst]
                frame_attr.texts = ocr_text_lst
                
                # Captions
                caption = caption_features[caption_features["n"] == frame]["captions"].values[0]
                frame_attr.caption = caption

                #
                frame_attrs.append(frame_attr.__dict__)
    
    np.save(os.path.join(args.path, "mapping.npy"), frame_attrs)
    


        