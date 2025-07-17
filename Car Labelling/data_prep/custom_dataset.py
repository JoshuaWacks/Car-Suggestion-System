from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os

class CustomDataset(Dataset):
    def __init__(self, parent_dir, df, transformations, label):
        self.parent_dir = parent_dir
        self.df = df
        self.transformations = transformations
        self.label = label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        img_folder = os.path.join(self.parent_dir, item.Original_Split)
        img_path = os.path.join(img_folder, item.Image)
        image = Image.open(img_path)
        tensor_image = self.transform(image)

        label = item[self.label]

        return tensor_image,label