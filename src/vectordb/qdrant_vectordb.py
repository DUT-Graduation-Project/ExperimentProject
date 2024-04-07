from .base import vectordb

#from qdrant_client import QdrantClient
#from qdrant_client.models import Distance, VectorParams

class QdrantVectordb(vectordb):
    def __init__(self, index_method : str = "cosine", feature_shape : int = 512, 
                 collection_name : str = "vectordb", storage_path : str = "./data/vectordb/qdrant"):
        self.collection_name = collection_name

        if index_method == "l2":
            vector_config = VectorParams(size = feature_shape, distance = Distance.EUCLID)
        elif index_method == "cosine":
            vector_config = VectorParams(size = feature_shape, distance = Distance.COSINE)
        else:
            assert f"{index_method} does not support"

        # Qdrant Client
        self.client = QdrantClient(storage_path)
        self.collection_name = collection_name

    def load(self):
        self.client.get_collection(
            collection_name= self.collection_name
        )
    
    def reset_index(self):
        return super().reset_index()
    
    def add(self):
        return super().add()

    def search(self):
        return super().search()
    
    def reconstruct(self):
        return super().reconstruct()
    
    def write_index(self):
        return super().write_index()
    