from abc import ABC, abstractmethod

class vectordb:
    @abstractmethod
    def __init__(self):
        return

    @abstractmethod
    def load(self):
        return

    @abstractmethod
    def reset_index(self):
        return
    
    @abstractmethod
    def add(self):
        return
    
    @abstractmethod
    def search(self):
        return
    
    @abstractmethod
    def reconstruct(self):
        return
    
    @abstractmethod
    def write_index(self):
        return