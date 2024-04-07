import os
import numpy as np
import argparse
import faiss

from reranking import average_qe, alpha_qe
from configs import get_args_parse

class GraphDataset:
    def __init__(self,args):
        self.args = args        
        self.rerank = alpha_qe(index_method="L2", alpha = 0.5)
        

    def build_img_similarity_qe(self):
        faiss_path = os.path.join(self.args.output_dir, "faiss_vit_32.bin")

        # Not Sure
        img_feats = faiss.read_index(faiss_path)

        img_D, img_I = img_feats.search(img_feats, self.args.top_k)
        
        np.save("/tmp/img_D", img_D)
        np.save("/tmp/img_I", img_I)
        
        img_feats_qe = self.rerank.query_expansion(img_feats, img_D, img_I, k = self.args.top_k)
        img_feats_qe /= np.linalg.norm(img_feats_qe, 2, axis=1, keepdims=True)

        img_feats = np.hstack([img_feats, img_feats_qe])
        img_feats /= np.linalg.norm(img_feats, axis=1).reshape((-1, 1))

        index = faiss.IndexFlatIP(img_feats.shape[1])

        index.add(img_feats)
        img_D, img_I = index.search(img_feats, self.args.top_k)

        np.save("/tmp/img_D_qe", img_D)
        np.save("/tmp/img_I_qe", img_I)

    def build_bert_feature(self):
        
        pass
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents = [get_args_parse()])
    
    args = parser.parse_args()
    
    dataset = GraphDataset(args)
    dataset.build_img_similarity_qe()