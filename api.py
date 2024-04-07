import uvicorn
import requests
import argparse
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from src.model_predictor import ModelPredictor
from src.configs import get_args_parse
from pydantic import BaseModel
from typing import List
import os

class ReturnedObject(BaseModel):
    title: str
    image_path: str


IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "submission")
SESSION_ID = os.getenv("HCMAI_SESSION_KEY")
URL = "https://eventretrieval.one/api/v1/submit"
SUBMIT_LIMIT = 5

# Đường dẫn tới thư mục public
public_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "public")
class AppAPI:
    def __init__(self, args: dict) -> None:
        self.app = FastAPI()
        self.app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")
        self.predictor = ModelPredictor(args)

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @self.app.get("/")
        async def root():
            return {"message": "hello"}

        @self.app.get("/text-search", response_model=List[ReturnedObject])
        async def text_search(query1: str, query2 : str, c : str, alpha : str):
            print(query1, query2)
            image_list = list(self.predictor.text_search(first_text=query1, 
                                                         second_text=query2,
                                                           c = int(c), 
                                                           alpha = float(alpha)))
            result_list = [
                ReturnedObject(title=image[1], image_path=image[0])
                for image in image_list
            ]

            return result_list

        @self.app.get("/image-search", response_model=List[ReturnedObject])
        async def image_search(image_path: str):
            image_list = list(self.predictor.image_search(image_path))
            result_list = [
                ReturnedObject(title=image[1], image_path=image[0])
                for image in image_list
            ]
            return result_list

        @self.app.post("/ocr-search", response_model=List[ReturnedObject])
        async def ocr_search(query: str, image_list: List[ReturnedObject]):
            url_list = [
                [image.title.split()[0], int(float(image.title.split()[1]))]
                for image in image_list
            ]
            image_list = self.predictor.ocr_search(query, url_list)
            result_list = [
                ReturnedObject(title=image[1], image_path=image[0])
                for image in image_list
            ]
            return result_list

        @self.app.get("/segment-search", response_model=List[ReturnedObject])
        async def segment_search(image_path: str):
            image_list = list(self.predictor.segment_search(image_path))
            result_list = [
                ReturnedObject(title=image[1], image_path=image[0])
                for image in image_list
            ]
            return result_list

        @self.app.post("/object-detection", response_model=List[ReturnedObject])
        async def object_detection(object: str, image_list: List[ReturnedObject]):
            url_list = [
                [image.title.split()[0], int(float(image.title.split()[1]))]
                for image in image_list
            ]
            image_list = list(self.predictor.object_detection_search(url_list, object))
            result_list = [
                ReturnedObject(title=image[1], image_path=image[0])
                for image in image_list
            ]
            return result_list

        @self.app.post("/submit")
        async def submit(submit_list: List[ReturnedObject]):
            for idx, sample in enumerate(submit_list):
                if idx == SUBMIT_LIMIT: break

                video, frame_idx = sample.title.split(" ")
                video = str(video)
                frame_idx = str(frame_idx)
                assert isinstance(video, str) and isinstance(frame_idx, str), "Input variables must be strings"

                PARAMS = {
                    "item" : video,
                    "frame" : frame_idx,
                    "session" : SESSION_ID
                }
                                
                response = requests.get(url = URL, params = PARAMS)
                print(response.content)

        
        @self.app.post("/submit-title")
        async def submit_single_image(data : ReturnedObject):
            title = data.title
            video, frame_idx = title.split(" ")
            video = str(video)
            frame_idx = str(frame_idx)
            assert isinstance(video, str) and isinstance(frame_idx, str), "Input variables must be strings"

            PARAMS = {
                "item" : video,
                "frame" : frame_idx,
                "session" : SESSION_ID
            }
            response = requests.get(url = URL, params = PARAMS)
            print(response.content)

    def run(self, port: int):
        uvicorn.run(self.app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[get_args_parse()])
    parser.add_argument("--faiss-name", type=str, default="faiss_vit_l_14_openai.bin")
    parser.add_argument("--model-name", type=str, default="ViT-L-14")
    parser.add_argument("--pretrain-model", type=str, default="openai")

    args = parser.parse_args()
    app = AppAPI(args)

    app.run(port=8000)
