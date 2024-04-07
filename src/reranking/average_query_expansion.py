import faiss
import numpy as np
import torch
import torch.nn.functional as F

from .base import rerank

class average_qe(rerank):
    def __init__(self, index_method : str = "l2"):
        super().__init__()
        self.index_method = index_method

    def db_augment(self, query_vecs, ref_vecs, k : int = 10):
        """
        Database-side feature augmentation (DBA)
        Albert Gordo, et al. "End-to-end Learning of Deep Visual Representations for Image Retrieval,"
        International Journal of Computer Vision. 2017.
        https://link.springer.com/article/10.1007/s11263-017-1016-8
        """
        query_vecs = query_vecs.astype(np.float32)
        ref_vecs = ref_vecs.astype(np.float32)

        weights = np.logspace(0, -2., k+1)

        # Query augmentation
        query_aug = faiss.IndexFlatIP(ref_vecs.shape[1])
        query_aug.add(ref_vecs)
        distances, indexes = query_aug.search(x=query_vecs, k= k)
        query_aug.reset()

        top_k_ref = ref_vecs[indexes]
        query_vecs = np.tensordot(weights, np.concatenate([np.expand_dims(query_vecs, 1), top_k_ref], axis=1), axes=(0, 1))

        # ref augmentation
        ref_aug = faiss.IndexFlatIP(ref_vecs.shape[1])
        ref_aug.add(ref_vecs)
        distances, indexes = ref_aug.search(x=ref_vecs, k=k  + 1)
        ref_aug.reset()

        top_k_ref = ref_vecs[indexes]
        ref_vecs = np.tensordot(weights, top_k_ref, axes=(0, 1))

        return query_vecs, ref_vecs

    def average_query_expansion(self, query_vecs, ref_vecs, k : int = 5):
        """
        Average Query Expansion (AQE)
        Ondrej Chum, et al. "Total Recall: Automatic Query Expansion with a Generative Feature Model for Object Retrieval,"
        International Conference of Computer Vision. 2007.
        https://www.robots.ox.ac.uk/~vgg/publications/papers/chum07b.pdf
        https://github.com/leeesangwon/PyTorch-Image-Retrieval/blob/public/inference.py
        """

        query_vecs = query_vecs.astype(np.float32)
        ref_vecs = ref_vecs.astype(np.float32)

        # Query augmentation
        query_aug = faiss.IndexFlatIP(ref_vecs.shape[1])
        query_aug.add(ref_vecs)
        distances, indexes = query_aug.search(x=query_vecs, k= k)
        query_aug.reset()

        top_k_ref_mean = np.mean(ref_vecs[indexes], axis=1, dtype=np.float32)
        query_vecs = np.concatenate([query_vecs, top_k_ref_mean], axis=1)

        # ref augmentation
        ref_aug = faiss.IndexFlatIP(ref_vecs.shape[1])
        ref_aug.add(ref_vecs)
        distances, indexes = ref_aug.search(x=ref_vecs, k=k  + 1)
        ref_aug.reset()

        top_k_ref_mean = np.mean(ref_vecs[indexes], axis=1, dtype=np.float32)
        ref_vecs = np.concatenate([ref_vecs, top_k_ref_mean], axis=1)

        return query_vecs, ref_vecs

    def retrieve(self, query_vecs, ref_vecs, ref_idx, k : int = 100):
        query_vecs, ref_vecs = self.db_augment(query_vecs=query_vecs,
                                               ref_vecs=ref_vecs)
        
        query_vecs, ref_vecs = self.average_query_expansion(query_vecs = query_vecs,
                                                            ref_vecs = ref_vecs)
        
        # Similarity Matrix after indexing processes (Feature Enhance (DBA,...), Rerank (QE, K-reciprocal,...))
        aug = faiss.IndexFlatIP(ref_vecs.shape[1])
        aug.add(ref_vecs)
        distances, rerank_indexes = aug.search(query_vecs, k = k)
        aug.reset()

        rerank_indexes = [[ref_idx[0][idx] for idx in rerank_indexes[0]]]
        
        return distances, rerank_indexes