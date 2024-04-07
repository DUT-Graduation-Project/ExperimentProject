import os
import argparse
import wandb
import fastdup
import pandas as pd
import shutil
import torch
import gc

from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import cv2
import numpy as np

from configs import get_args_parse
from utils import WandbLogger

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large"
)

INPUT_DIR = "/kaggle/hcmai-keyframe/keyframe/"
OUTPUT_DIR = "/kaggle/working/"


def generate_blip_labels(filenames, kwargs):
    # print('got files', filenames)
    try:
        preds = []
        images = []
        for image_path in filenames:
            i_image = Image.open(image_path)
            if i_image is not None:
                i_image = cv2.cvtColor(np.array(i_image), cv2.COLOR_BGR2RGB)
                im_pil = Image.fromarray(i_image)
                images.append(im_pil)
            else:
                print("Non image" + image_path)

        inputs = processor(images, return_tensors="pt")
        out = model.generate(**inputs)
        for i in range(len(out)):
            preds.append((processor.decode(out[i], skip_special_tokens=True)))
        return preds
    except Exception as e:
        print(e)
        # fastdup_capture_exception("Auto caption image blip", e)
        return None


def image_captioning(url_lst: list, key_num: str, batch_size: int = 64):
    image_urls = []
    captions = []

    for i in range(len(url_lst) // batch_size):
        curfiles = url_lst[i * batch_size : i * batch_size + batch_size]
        curout = generate_blip_labels(curfiles, {})
        image_urls.extend(curfiles)
        captions.extend(curout)

    os.makedirs(f"/kaggle/working/{key_num}/", exist_ok=True)

    vid = url_lst[0].split("/")[-2]

    df = pd.DataFrame({"n": image_urls, "captions": captions})
    df.to_csv(f"/kaggle/working/{key_num}/{vid}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[get_args_parse()])

    args = parser.parse_args()

    wandb.login(key=secret_value_0)
    run = wandb.init(project="hcmai")

    excludes = []
    for key_num in tqdm(os.listdir(INPUT_DIR)):
        if key_num not in ["keyframes-" + num for num in excludes]:
            for vid in tqdm(os.listdir(os.path.join(INPUT_DIR, key_num))):
                url_lst = os.path.join(INPUT_DIR, key_num, vid)

                image_captioning(url_lst, key_num=key_num, batch_size=args.batch_size)

        shutil.make_archive(
            os.path.join(OUTPUT_DIR, key_num),
            "zip",
            os.path.join(OUTPUT_DIR, key_num),
        )
        # Wandb
        artifact = wandb.Artifact(name="image_caption", type="dataset")
        artifact.add_file(os.path.join(OUTPUT_DIR, key_num) + ".zip")
        run.log_artifact(artifact)

        gc.collect()
        torch.cuda.empty_cache()
