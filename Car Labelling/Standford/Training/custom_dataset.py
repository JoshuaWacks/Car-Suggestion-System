from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torch


class CustomDataset(Dataset):
    def __init__(self, parent_dir, df, transformations, label, label_descr):
        self.parent_dir = parent_dir
        self.df = df
        self.transformations = transformations
        self.label = label
        self.label_descr = label_descr
        self.class_values = sorted(df[label_descr].unique())

    def __len__(self):
        return len(self.df)
    
    def classes(self, idx):
        return self.class_values[idx]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
                    idx = idx.tolist()

        item = self.df.iloc[idx]
        img_split = os.path.join(self.parent_dir, item.Original_Split)
        img_folder = os.path.join(img_split, item.Full_Name)
        img_path = os.path.join(img_folder, item.Image)
        image = Image.open(img_path)
        tensor_image = self.transformations(image)

        label = item[self.label]

        return tensor_image,label
        