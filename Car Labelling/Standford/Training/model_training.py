import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from sklearn.metrics import f1_score,precision_score,recall_score
from PIL import Image
from tempfile import TemporaryDirectory
import mlflow
from mlflow.models import infer_signature
import GPUtil
from torch.utils.data import  DataLoader


from utils import Utils
from custom_dataset import CustomDataset


class ModelTraining():

    def __init__(self, parent_dir, full_df_path, params, model):

        self.utils = Utils()

        self.parent_dir = parent_dir
        self.df = pd.read_csv(full_df_path)
        self.params = params
        self.model = model

        self.train_transforms, self.val_transforms = self.utils.get_transformations(params)
        self.train_dataset, self.val_dataset, self.test_dataset = self.__get_datasets()
        self.train_loader, self.val_loader, self.test_loader = self.__get_dataloaders()

    def __get_datasets(self):
        dataset_info = self.params['dataset_info']
        if dataset_info['pre_cropped']:
            self.df = self.df[self.df['PreCropped'] == True]
            self.parent_dir = os.path.join(self.parent_dir,'cropped')
        else:
            self.df = self.df[self.df['PreCropped'] == False]
            self.parent_dir = os.path.join(self.parent_dir,'original')

        self.train, self.validate, self.test = np.split(self.df.sample(frac=1, random_state=self.params['random_state']), 
                       [int(.6*len(self.df)), int(.8*len(self.df))])
        
        train_dataset = CustomDataset(parent_dir=self.parent_dir,
                                    df = self.train,
                                    transformations=self.train_transforms, 
                                    label = dataset_info['label'],
                                    label_descr=dataset_info['label_descr'])
        
        val_dataset = CustomDataset(parent_dir= self.parent_dir,
                                    df = self.validate,
                                    transformations=self.val_transforms, 
                                    label = dataset_info['label'],
                                    label_descr=dataset_info['label_descr'])
        
        test_dataset = CustomDataset(parent_dir=self.parent_dir,
                                    df = self.test,
                                    transformations=self.val_transforms, 
                                    label = dataset_info['label'],
                                    label_descr=dataset_info['label_descr'])
            
        return train_dataset, val_dataset, test_dataset 
    
    def __get_dataloaders(self):
        training_params = self.params['training_params']

        train_loader = DataLoader(self.train_dataset, 
                    batch_size=training_params['batch_size'], 
                    shuffle=training_params['shuffle'], 
                    num_workers=training_params['num_workers'],
                    pin_memory=training_params['pin_memory'])
        
        val_loader = DataLoader(self.val_dataset, 
            batch_size=training_params['batch_size'], 
            shuffle=training_params['shuffle'], 
            num_workers=training_params['num_workers'],
            pin_memory=training_params['pin_memory'])
        
        test_loader = DataLoader(self.test_dataset, 
            batch_size=training_params['batch_size'], 
            shuffle=training_params['shuffle'], 
            num_workers=training_params['num_workers'],
            pin_memory=training_params['pin_memory'])
        
        return train_loader, val_loader, test_loader
        
