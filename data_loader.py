import os
import numpy as np

class DataLoaderFor4s:
    def __init__(self, target_volume):
        self.target_volume: str = target_volume # The kind of organ or lesion. (str)
        self.volume_id_list: list = [] # The length of this list is corresponding to the total number of organ or lesion.
        
        self.volumes: list = [] # テンソル型に変換するのは呼び出された後でよい？
    
    def generate_id_list(self, target_name):
        id_list: list = []
        return id_list
    
    def generate_volumes(self, id_list):
        volumes: list = []
        return volumes
    
    def id_to_volume(self, patient_id):
        return

class DataLoaderFor2D:
    def __init_(self, target_volume):
        self.target_volume: str = target_volume # The kind of organ or lesion. (str)