import torch
import open_clip
import argparse
import faiss
import numpy as np
import json
from PIL import Image
import pandas as pd
import logging
from tqdm import tqdm

import utils
from configs import get_args_parse
from vectordb import FaissVectordb
torch.manual_seed(0)

class ClipDatabase:
    def __init__(self, args):
        self.device = args.device
        self.model, self.preprocess = open_clip.create_model_from_pretrained(args.model_name, 
                                                                             args.pretrain_model,
                                                                             )

        # Reproducibility
        self.model.to(self.device)
        self.model.eval()
        
        self.index = FaissVectordb(index_method= args.index_method, feature_shape=args.feature_shape)

        self.json_object = {} 
        self.feature_shape = args.feature_shape
        self.batch_size = args.batch_size

    @torch.no_grad()
    def get_text_features(self, text: str):
        token = open_clip.tokenize([text]).to(self.device)
        text_features = self.model.encode_text(token)
        return text_features.detach().cpu().numpy()
    
    @torch.no_grad()
    def preprocess_image(self, image : list):
        image = [self.preprocess(Image.open(path)) for path in image]
        image_features = self.model.encode_image(torch.stack(image).to(self.device))
        return image_features.detach().cpu().numpy()

    def create_mapping(self, image_path: str, id: int, vid : str, frame_idx : int):
        self.json_object[id] = {
            "image_path" : image_path, 
            "video" : vid, 
            "frame_idx" : frame_idx
        }
    
    @utils.fcall
    def load_database(self, faiss_dir : str):
        self.index = faiss.loader()
        return database
    
    @utils.fcall
    def create_image_database(self, dir : str, output_dir : str):
        logging.info('Start to create the Database')
        count = 0

        
        img_npy = np.load(dir, allow_pickle=True)
        img_df = pd.DataFrame.from_records(img_npy)
            
        image_lst = img_df["filepath"]
        vid_lst = img_df["vid_src"]
        frame_idx_lst = img_df["frame_idx"]
        
        # Create Embedding
        embedding_vectors = []
        for idx in tqdm(range(0, len(image_lst), self.batch_size), position = 0, leave = True):
            images = [image_lst[id] for id in range(idx, idx + self.batch_size)]
            embeddings = self.preprocess_image(images)
            for embedding in embeddings:
                embedding_vectors.append(embedding)
        
        images = image_lst[idx : idx + self.batch_size] 
        embeddings = self.preprocess_image(images)
        for embedding in embeddings:
                embedding_vectors.append(embedding)

        # Build Faiss VectorDatabase
        for vector, image_path, vid, frame_idx in tqdm(zip(embedding_vectors, image_lst, vid_lst, frame_idx_lst), position = 0, leave=True):
            self.index.add(vector)
            self.create_mapping(image_path, count, vid, frame_idx)
            count += 1

        logging.info("Total images: {} images.".format(count))

        self.index.write_index(output_dir = f"{output_dir}faiss_img_vit_l_14_openai.bin")

        with open(f"{output_dir}/mapping.json", "w") as f:
            json.dump(self.json_object, f)

    @utils.fcall
    def create_caption_database(self, input_dir : str, output_dir : str):
        logging.info('Start to create the Database')
        count = 0

        
        npy = np.load(input_dir, allow_pickle=True)
        df = pd.DataFrame.from_records(npy)
        captions = df["caption"]
        
        embedding_vectors = [self.get_text_features(caption) for caption in tqdm(captions)]

        # Build Faiss VectorDatabase
        for vector in tqdm(embedding_vectors, position = 0, leave=True):
            self.index.add(vector)
            count += 1

        logging.info("Total caption: {} captions.".format(count))

        self.index.write_index(output_dir = f"{output_dir}faiss_caption_vit_l_14_openai.bin")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Main Configuration", parents=[get_args_parse()])
    parser.add_argument("--run-name", type = str, default = "clip_database")
    parser.add_argument("--batch-size", type = int, default = 32)
    parser.add_argument("--model-name", type=str, default="ViT-L-14")
    parser.add_argument("--pretrain-model", type = str, default = "openai")
    args = parser.parse_args()

    #utils.setup_logging()
    
    database = ClipDatabase(
        args = args
    )
    
    if args.run_name == "clip_database":
        database.create_image_database(args.mapping_dir, args.output_dir)
    elif args.run_name == "caption_database":
        database.create_caption_database(args.mapping_dir, args.output_dir)
    else:
        assert "Run name can not be recognized"
    
    if args.enable_wandb:
        wandb_logger = utils.WandbLogger(args)
        wandb_logger.log_vectordb(args.output_dir)
