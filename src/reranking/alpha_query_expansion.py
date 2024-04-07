import numpy as np
import faiss

from .base import rerank

class alpha_qe(rerank):
    def __init__(self, index_method : str = "cosine", alpha : int = 0.5):
        self.index_method = index_method
        self.alpha = alpha

    def query_expansion(self, vecs, sims, idx, k: int = 2):
        weights = np.expand_dims(sims[:, :k] ** self.alpha, axis=-1).astype(np.float64)
        vecs = (vecs[idx[:, :k]] * weights).sum(axis=1)
        return vecs
    def alpha_query_expansion(self, query_vecs, ref_vecs, k: int = 100):
        '''
        Alpha query expansion
        Radenovic, F., Tolias, G., Chum, O.: Fine-tuning cnn image retrieval with no human
        annotation. TPAMI (2018)
        '''
        query_vecs = query_vecs.astype(np.float64)
        ref_vecs = ref_vecs.astype(np.float64)

        # Query augmentation        
        combined_vecs = np.concatenate((query_vecs, ref_vecs))

        query_aug = faiss.IndexFlatIP(ref_vecs.shape[1])
        query_aug.add(combined_vecs)
        distances, indexes = query_aug.search(x= combined_vecs, k= k)
        query_aug.reset()

        feats_qe = self.query_expansion(combined_vecs, distances, indexes)
        feats_qe /= np.linalg.norm(feats_qe, 2, axis=1, keepdims=True)

        feats = np.hstack([combined_vecs, feats_qe])
        feats /= np.linalg.norm(feats, axis=1).reshape((-1, 1))

        query_vecs = feats[0,:].reshape((1,-1))
        ref_vecs = feats[1:]

        return query_vecs, ref_vecs
    
    def retrieve(self, query_vecs, ref_vecs, ref_idx, k : int = 100):        
        query_vecs, ref_vecs = self.alpha_query_expansion(query_vecs = query_vecs,
                                                            ref_vecs = ref_vecs)
        
        # Similarity Matrix after indexing processes (Feature Enhance (DBA,...), Rerank (QE, K-reciprocal,...))
        aug = faiss.IndexFlatIP(ref_vecs.shape[1])
        aug.add(ref_vecs)
        distances, rerank_indexes = aug.search(query_vecs, k = k)
        aug.reset()

        rerank_indexes = [[ref_idx[0][idx] for idx in rerank_indexes[0]]]

        return distances, rerank_indexes