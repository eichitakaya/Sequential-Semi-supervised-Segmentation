import os
import numpy as np
import torch
import nibabel as nib
import glob

class DatasetForSemi(torch.utils.data.Dataset):
    # input: データセット名（ex:heart, lung cancer, etc）
    # output: 1セットの連続画像を吐き出す．
    # memo: 基本的にシャッフルはしない（する必要がない）
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        
    
class DataLoaderFor4s:
    def __init__(self, target_volume):
        self.target_volume_path: str = get_dataset_name(target_volume) # The kind of organ or lesion. (str)
        self.patient_id_list: list = get_patient_id_list()
        #self.volume_id_list: list = get_volumes() # The length of this list is corresponding to the total number of organ or lesion.
        
        self.volumes: list = [] # テンソル型に変換するのは呼び出された後でよい？
    
    def get_patient_id_list(self):
        path = self.target_volume
        id_list = glob.glob(path + "*.nii*")
        return id_list
    
    def get_volumes(self, patient_id):
        # 全てのラベルが連続している（真っ黒な画像を挟まない）症例のみを抽出
        # とりあえず，heart, spleen, hippocampusは連続していることにしておく
        # １症例に複数の病変が含まれる場合については，後回し
        
        volumes: list = glob.glob()
        return volumes
    
    def id_to_volumes(self, patient_id):
        # 全てのラベルが連続している（真っ黒な画像を挟まない）症例のみを抽出
        # とりあえず，heart, spleen, hippocampusは連続していることにしておく
        # １症例に複数の病変が含まれる場合については，後回し
        # __getitem__の中で使う
        return
    
    def __getitem__(self, item):
        patient_id = self.patient_id_list[item]
        return id_to_volumes(patient_id)

class GroupWiseSplit():
    def __init__(self, target_volume, val_ratio, group_shuffle):
        self.target_volume: str = target_volume # Choose organ or lesion. (str)
        self.val_ratio: float = val_ratio
        self.shuffle: bool = group_shuffle
        
        self.base_path = get_dataset_name[target_volume.lower()]
            
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


def get_dataset_name(name):
    name_path_dict = {"hippocampus":"/takaya_workspace/Medical_AI/data/decathlon/Task04_Hippocampus/imagesTr/",
                      "heart":"/takaya_workspace/Medical_AI/data/decathlon/Task02_Heart/imagesTr/",
                      "spleen":"/takaya_workspace/Medical_AI/data/decathlon/Task09_Spleen/imagesTr/"}
    return name_path_dict[name]

if __name__ == "__main__":
    data = DataLoaderFor4s("heart")
    print(data.target_volume)