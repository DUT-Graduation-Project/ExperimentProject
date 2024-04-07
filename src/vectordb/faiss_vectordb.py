import faiss
import numpy as np

from .base import vectordb

class FaissVectordb(vectordb):
    def __init__(self, index_method : str = "cosine", feature_shape : int = 512):
        if index_method == "l2":
            self.index = faiss.IndexFlatL2(feature_shape)
        elif index_method == "cosine":
            self.index = faiss.IndexFlatIP(feature_shape)
        else:
            assert f"{index_method} does not support"

        self.index.n_probe = 100

    def add(self, vector):
        self.index.add(vector)

    def read_index(self, input_dir : str):    
        self.index = faiss.read_index(input_dir)

    def search(self, query_vecs , top_k : int):
        distances, indexes = self.index.search(query_vecs, top_k)

        return distances, indexes
    
    def reconstruct(self, indexes):
        return self.index.reconstruct_batch(indexes)
    
    def get_embedding(self):
        ntotal = self.index.ntotal
        feature_shape = self.index.d

        return faiss.rev_swig_ptr(self.index.get_xb(), ntotal * feature_shape).reshape((ntotal, feature_shape)) 
    
    def reset_index(self):
        self.index.reset()
    
    def set_param(self, **kwargs):
        for arg in kwargs:
            self.index.arg = kwargs[arg]
        
    def write_index(self, output_dir : str):
        faiss.write_index(self.index, output_dir)

