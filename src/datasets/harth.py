import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
from torchvision import transforms, datasets
from torchvision.models import resnet18, resnet34, resnet50
from tqdm import tqdm
import os
import pandas as pd
import einops
from scipy.stats import mode
import numpy as np


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def complex_to_cartesian(x):
    return torch.stack([torch.real(x), torch.imag(x)], -1)

def complex_to_magnitude(x, expand=False):
    magnitude = torch.abs(x)
    return torch.unsqueeze(magnitude, -1) if expand else magnitude



    

class HarthDownstream(Dataset):
    def __init__(self, data_path, window_size):
        
        # Usage:
        self.unique_labels = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 130, 140])
        
        tensor_data, label_data = self.read_files(data_path)
        
        remaining = len(tensor_data) % window_size
        if remaining > 0:
            tensor_data = tensor_data[:-remaining]
            label_data = label_data[:-remaining]
            
        
        
        self.tensor_data = torch.reshape(tensor_data, (len(tensor_data)//window_size, window_size, -1))
        self.label_data = torch.reshape(label_data, (len(label_data)//window_size, window_size, -1))
        
        
    def __len__(self):
        return len(self.tensor_data)

    def __getitem__(self, index):
        return self.tensor_data[index], self.label_data[index]
    
    def mode_sliding_window(self, data, window_size, stride):
        modes = []
        for i in range(0, len(data) - window_size + 1, stride):
            window = data[i:i + window_size]
            window_mode = mode(window)[0]
            modes.append(window_mode)
        return torch.tensor(np.array(modes)).squeeze()
    
    def convert_to_class_indices(self, original_labels):
        class_indices = torch.zeros_like(original_labels)
        for i, label in enumerate(original_labels):
            class_indices[i] = (self.unique_labels == label).nonzero().item()
        return class_indices

    
    def read_files(self, file_path):
        
        """ Reads all csv files in a given path"""
        data = []
        data_labels = []
        sum_1 = 0
        filenames = [x for x in os.listdir(file_path)]

        for fn in tqdm(filenames):
            df = pd.read_csv(
                os.path.join(file_path, fn),
            )

            x = torch.tensor(df.iloc[:, 1:7].values,dtype=torch.float32)
            labels = df[['label']].values
            
            x = einops.rearrange(x, 'S C -> C S')

            x = torch.stft(
                        input=x,
                        n_fft=150,
                        hop_length=75,
                        win_length=150,
                        window=torch.hann_window(150),
                        center=False,
                        return_complex=True
                    )  # [num_channels, num_bins, num_frames]
            x_cartesian = complex_to_cartesian(x)
            x_magnitude = complex_to_magnitude(x, expand=True)
        #     x = x_cartesian
            x = x_magnitude

        #     x = einops.rearrange(x, 'C F T P -> T (C F P)')  # P=2

            x = einops.rearrange(x, 'C F T P -> T C F P')
            
            # usage:
            window_size = 150
            stride = 75

            processed_labels = self.mode_sliding_window(labels, window_size, stride)

            data.append(x)
            data_labels.extend(self.convert_to_class_indices(processed_labels))
        
    
        new_data = torch.cat(data, dim=0).squeeze()
        new_data = new_data.reshape(-1, new_data.shape[0]).T
    
        data_labels = torch.tensor(data_labels)
        new_labels = data_labels.reshape(-1, data_labels.shape[0]).T

        return new_data, new_labels
    

class SequentialRandomSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size


    def __iter__(self):
        

        indices = list(range(len(self.data_source)))
        
        remaining = len(indices) % self.batch_size
        if remaining > 0:
            indices = indices[:-remaining]
        final_indices = np.reshape(indices, (-1, self.batch_size))

        # Shuffle the batches
        np.random.shuffle(final_indices)

        # Flatten the list of batches to get the final order of indices
        final_indices = [(idx, idx) for batch in final_indices for idx in batch]
        
        return iter(final_indices)

    def __len__(self):
        return len(self.data_source)

class HarDataset(Dataset):
    def __init__(self, data_path, window_size):
        
        data = self.read_files(data_path)
        
        remaining = len(data) % window_size
        if remaining > 0:
            data = data[:-remaining]
        
        self.final_data = torch.reshape(data, (len(data)//window_size, window_size, -1))
        
    def __len__(self):
        return len(self.final_data)

    def __getitem__(self, index):
        return self.final_data[index[0]]
    
    def read_files(self, file_path):
        
        """ Reads all csv files in a given path"""
        data = []
        filenames = [x for x in os.listdir(file_path)]

        for fn in tqdm(filenames):
            df = pd.read_csv(
                os.path.join(file_path, fn),
            )

            x = torch.tensor(df.iloc[:, 1:7].values,dtype=torch.float32)

            x = einops.rearrange(x, 'S C -> C S')

            x = torch.stft(
                        input=x,
                        n_fft=50,
                        hop_length=25,
                        win_length=50,
                        window=torch.hann_window(50),
                        center=False,
                        return_complex=True
                    )  # [num_channels, num_bins, num_frames]
            x_cartesian = complex_to_cartesian(x)
            x_magnitude = complex_to_magnitude(x, expand=True)
        #     x = x_cartesian
            x = x_magnitude

        #     x = einops.rearrange(x, 'C F T P -> T (C F P)')  # P=2

            x = einops.rearrange(x, 'C F T P -> T C F P')

            data.append(x)

        new_data = torch.cat(data, dim=0).squeeze()
        new_data = new_data.reshape(-1, new_data.shape[0]).T

        return new_data
    
    class SequentialRandomSampler2(torch.utils.data.Sampler):
        def __init__(self, data_source, batch_size):
            self.data_source = data_source
            self.batch_size = batch_size


        def __iter__(self):
            

            indices = list(range(len(self.data_source)))
            
            remaining = len(indices) % self.batch_size
            if remaining > 0:
                indices = indices[:-remaining]
            final_indices = np.reshape(indices, (-1, self.batch_size))

            # Shuffle the batches
            np.random.shuffle(final_indices)

            # Flatten the list of batches to get the final order of indices
            final_indices = [idx for batch in final_indices for idx in batch]
            
            return iter(final_indices)

        def __len__(self):
            return len(self.data_source)