import numpy as np
import faiss
from .base import rerank

class attention_qe(rerank):
    def __init__(self, index_method : str = "l2"):
        self.index_method = index_method

    def attention_query_expansion(self, query_vecs, ref_vecs):
        
        return query_vecs, ref_vecs
    
    def retrieve(self, query_vecs, ref_vecs):

        return query_vecs, ref_vecs