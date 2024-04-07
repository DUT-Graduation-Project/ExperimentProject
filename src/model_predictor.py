import os
import argparse
import json
import numpy as np
import pandas as pd

import open_clip
from typing import Any
import torch
from PIL import Image

from .vectordb import FaissVectordb
from .reranking import average_qe, alpha_qe, Reranking1_Shoppe
from .text_processing import Translator
from .ocr_processing import get_ocr_results
from .configs import get_args_parse
from .utils import *


class ModelPredictor:
    def __init__(self, args: dict):
        self.args = args
        self.model, self.preprocess = open_clip.create_model_from_pretrained(args.model_name, args.pretrain_model)

        # Reproducibility
        self.model.eval()
        
        if args.vectordb == "faiss":
            self.faiss_index = FaissVectordb(args.index_method, feature_shape=args.feature_shape)

            faiss_path = os.path.join(args.output_dir, args.faiss_name)
            self.faiss_index.read_index(input_dir= faiss_path)
        
        # mapping
        mapping_path = os.path.join(args.output_dir, "mapping.json")
        self.json_mapping = pd.read_json(mapping_path).T.sort_values(by = ["video", "frame_idx"])
            
        # Resources
        self.resources = np.load(args.mapping_dir, allow_pickle = True)
        self.resources = pd.DataFrame.from_records(self.resources)
        
        # Translator
        self.translator = Translator()

        if args.rerank == "aqe":
            self.rerank = average_qe(index_method=args.index_method)
        elif args.rerank == "alpha_qe":
            self.rerank = alpha_qe(index_method=args.index_method, alpha=0.5)
        elif args.rerank == "shoppee":
            self.rerank = Reranking1_Shoppe(index_method=args.index_method)
        else:
            self.rerank = None

        if args.temporal_search == "brute_force":
            self.temporal_search = None
        else:
            self.temporal_search = None

        # 
        self.k = args.k
        self.top_k = args.top_k

    @torch.no_grad()
    def get_image_features(self, image_path: str):
        image = self.preprocess(Image.open(image_path)).unsqueeze(0)
        image_features = self.model.encode_image(image)
        return image_features.detach().cpu().numpy()

    @torch.no_grad()
    def get_text_features(self, text: str):
        token = open_clip.tokenize([text])
        text_features = self.model.encode_text(token)
        return text_features.detach().cpu().numpy()

    def get_path_and_title(self, input_lst: list):
        paths = []
        titles = []
        for index in input_lst:
            sample = self.json_mapping.loc[index]
            img_path = sample["image_path"]

            title = " ".join([str(sample["video"]), str(int(sample["frame_idx"]))])

            paths.append(img_path)
            titles.append(title)

        return zip(paths, titles)
    
    @fcall
    def text_search(self, first_text: str, second_text : str = "", c = 5, alpha = 1.0):
        '''
            Text Search
            Input:
                - Text : str
            Ouput:
                - Path and title of Images: zip(list, list)
        '''
        #second_text = "cảnh một vụ nổ bên đường."

        # Text search with first query
        first_eng_text = self.translator(first_text)
        print(first_eng_text)
        first_query_vecs = self.get_text_features(first_eng_text)

        # Compute Simlarity Score and get ordered indexes of outputs
        FirstScore , first_ref_indexes = self.faiss_index.search(first_query_vecs, self.k)
        first_ref_vecs = self.faiss_index.reconstruct(first_ref_indexes[0])

        # Similarity Score and indexes of output after Reranking
        if self.rerank is not None:
            FirstScore , first_rerank_indexes = self.rerank.retrieve(query_vecs = first_query_vecs, 
                                                      ref_vecs = first_ref_vecs, 
                                                      ref_idx = first_ref_indexes, 
                                                      k = self.top_k)
            
            output_indexes = first_rerank_indexes[0]
        else:
            output_indexes = first_ref_indexes[0]
        
        # Temporal search with second text query
        if second_text != "" and alpha != 1.0:
            second_eng_text = self.translator(second_text)
            print(second_eng_text)
            second_query_vecs = self.get_text_features(second_eng_text)
            TotalScore = [[FirstScore[0][idx], output_indexes[idx]] for idx in range(len(FirstScore[0]))]


            for first_index, frame_idx in enumerate(output_indexes):
                sample_idx = self.json_mapping.index.get_loc(frame_idx)
                         
                next_c_idxs = [np.asarray([int(self.json_mapping.iloc[sample_idx + idx + 1].name) for idx in range(c) if len(self.json_mapping) > sample_idx + idx + 1])]
                next_c_vecs = self.faiss_index.reconstruct(next_c_idxs[0])
                
                second_faiss_idx = FaissVectordb(index_method=self.args.index_method, feature_shape = self.args.feature_shape)
                second_faiss_idx.add(next_c_vecs)

                MaxSecondScore , _ = second_faiss_idx.search(second_query_vecs, top_k = c)
                                    
                TotalScore[first_index][0] = (alpha) * (TotalScore[first_index][0]) * (1 - alpha) * (MaxSecondScore[0][0])

                second_faiss_idx.reset_index()

            # Sort total score
            TotalScore = sorted(TotalScore, key = lambda x: x[0], reverse = True)
            output_indexes = [sample[1] for sample in TotalScore]

        output = self.get_path_and_title(output_indexes)            
        return output
    
    @fcall
    def image_search(self, image_path: str):

        image_features = self.get_image_features(image_path)
        scores, indexes = self.faiss_index.search(image_features, self.k)

        ref_vecs = self.faiss_index.reconstruct(indexes[0])

        if self.rerank is not None:
            _ , rerank_indexes = self.rerank.retrieve(query_vecs = image_features, 
                                                      ref_vecs = ref_vecs, 
                                                      ref_idx = indexes, 
                                                      k = self.top_k)
            output_indexes = rerank_indexes[0]
        else:
            output_indexes = indexes[0]
        
        
        output = self.get_path_and_title(output_indexes)      
        return output
    
    @fcall
    def segment_search(self, image_path : str):
        vid_src = image_path.split("/")[-2]
        df = self.resources[self.resources["vid_src"] == vid_src]
        frame_idx = df[df["filepath"] == image_path]["frame_idx"].values[0]
        
        df_sorted = df.sort_values(by = "frame_idx", key = lambda x : abs(x - frame_idx), ascending=True)
        
        imgs = df_sorted["filepath"].values[:self.top_k]
        title = df_sorted.apply(lambda x : str(x["vid_src"]) + " " + str(int(x["frame_idx"])), axis = 1).values[:self.top_k]
        
        return zip(imgs, title)
    
    @fcall
    def ocr_search(self, ocr_text: str, url_lst : list):
        vid_src = [url[0] for url in url_lst]
        frame_idx = [url[1] for url in url_lst]
        df = self.resources[self.resources['vid_src'].isin(vid_src)]
        df = df[df["frame_idx"].isin(frame_idx)]

        df = df[df["texts"].apply(lambda x: get_ocr_results(ocr_text, x))]

        imgs = df["filepath"].values[:self.top_k]
        title = df.apply(lambda x : str(x["vid_src"]) + " " + str(int(x["frame_idx"])), axis = 1).values[:self.top_k]

        return zip(imgs, title)
    
    @fcall
    def object_detection_search(self, url_results : list, objects : str):
        vid_src = [url[0] for url in url_results]
        frame_idx = [float(url[1]) for url in url_results]
        df = self.resources[self.resources["vid_src"].isin(vid_src)]
        df = df[df["frame_idx"].isin(frame_idx)]

        df = df[df["objects"].apply(lambda x: objects in x)]

        imgs = df["filepath"].values[: self.top_k]
        title = df.apply(
            lambda x: str(x["vid_src"]) + " " + str(int(x["frame_idx"])), axis=1
        ).values[: self.top_k]
        return zip(imgs, title)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Main Configuration", parents=[get_args_parse()])
    parser.add_argument("--faiss-img-name", type=str, default="faiss_img_vit_l_14_openai.bin")
    parser.add_argument("--faiss-caption-name", type = str, default="faiss_caption_vit_l_14_openai.bin")
    parser.add_argument("--model-name", type=str, default="ViT-L-14")
    parser.add_argument("--pretrain-model", type=str, default="openai")

    args = parser.parse_args()

    model_predictor = ModelPredictor(
        args=args,
    )

    print(model_predictor.text_search("con chó vừa béo vừa kiêu", "con chó"))
    #print(database_processing.image_search("./data/images/imgs/img_0_0_0.jpg"))