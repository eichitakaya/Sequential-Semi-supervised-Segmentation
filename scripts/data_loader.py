import os
import numpy as np
import torch
import nibabel as nib
import glob

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

class GroupWiseSplit():
    def __init__(self, target_volume, val_ratio, group_shuffle):
        self.target_volume: str = target_volume # Choose organ or lesion. (str)
        self.val_ratio: float = val_ratio
        self.shuffle: bool = group_shuffle
            
        if target_volume.lower() == "hippocampus":
            self.base_path = "../data/Task04_Hippocampus/imagesTr/"
        if target_volume.lower() == "heart":
            self.base_path = "../data/Task02_Heart/imagesTr/"
        if target_volume.lower() == "spleen":
            self.base_path = "../data/Task09_Spleen/imagesTr/"
            
        self.groups: list = self._get_groups(self.base_path)
        
        self.train_groups: list
        self.val_groups: list
        self.train_groups, self.val_groups = self._split(self.groups, self.val_ratio)
        
    def _get_groups(self, base_path) -> list:
        groups = glob.glob(base_path + "*.nii*")
        return groups
        
    def _split(self, groups, val_ratio) -> list:
        val_n: int = int(len(groups) * val_ratio)
        train_n: int = len(groups) - val_n
        train_groups: list = groups[:train_n]
        val_groups: list = groups[train_n:train_n+val_n]
        return train_groups, val_groups
    
class DataSetFor2DSegmentation(torch.utils.data.Dataset):
    def __init__(self, groups, target_focus=False, transform=None, train=True):
        self.groups: list = groups
        self.target_focus = target_focus
        self.transform = transform
        
        self.train_flag = train
        
        self.data: list = []
        self.label: list = []
        self.data, self.label = self._set_array(self.groups, target_focus)
    
    def __len__(self) -> str:
        return len(self.data) 
    
    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_label = self.label[idx]
        
        if self.transform:
            out_data = self.transform(out_data)
            out_label = self.transform(out_label)
        
        return out_data, out_label
    
    def _set_array(self, groups, target_focus):
        data = []
        label = []
        
        for group in groups:
            arr_data = nib.load(group).get_data()
            arr_data = arr_data.transpose(2, 0, 1)
            group = group.replace("images", "labels")
            arr_label = nib.load(group).get_data()
            arr_label = arr_label.transpose(2, 0, 1)
            data.append(arr_data)
            label.append(arr_label)
        
        data = np.concatenate(data, 0)
        label = np.concatenate(label, 0)
        if target_focus:
            focus_data = []
            focus_label = []
            for i in range(len(data)):
                if label[i].max() > 0:
                    focus_data.append(data[i])
                    focus_label.append(label[i])
            data = focus_data
            label = focus_label
        return data, label
    
    
        
class DataLoaderFor3D(torch.utils.data.Dataset):
    def __init__(self, target_volume):
        self.target_volume: str = target_volume # Choose organ or lesion.
    
    def __len__(self):
        #total number of lesions(patients)
        return 
            
    
            
class Decathlon(torch.utils.data.Dataset):
    def __init__(self, transform = None):
        self.transform = transform
        
    def __len__(self):
        #total number of slices
        return 