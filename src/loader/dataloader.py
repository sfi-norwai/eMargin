import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
import pickle
import src.config, src.utils
import einops
import pandas as pd



# Function to load a pickle file
def load_pickle_file(filepath):
    with open(filepath, 'rb') as file:
        data = pickle.load(file)
    return data

def feature_stft(x_train, n_fft = 100, hop_length=50, win_length=100, phase = False, stack_axes = True):
   
    train_tensor = torch.tensor(x_train).transpose(1,2).reshape(-1, 2).transpose(0,1)

    x = torch.stft(
        input=train_tensor,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=torch.hann_window(win_length),
        center=False,
        return_complex=True)  # [num_channels, num_bins, num_frames]

    x_cartesian = src.utils.complex_to_cartesian(x)
    x_magnitude = src.utils.complex_to_magnitude(x, expand=True)

    x = x_cartesian if phase else x_magnitude
    if stack_axes:
        # Stack all spectrograms and put time dim first:
        # [num_channels, num_bins, num_frames, stft_parts] ->
        # [num_frames, num_channels x num_bins x stft_parts]
        x = einops.rearrange(x, 'C F T P -> T (C F P)')  # P=2
    else:
        x = einops.rearrange(x, 'C F T P -> T C F P')
    
    return x

def feature_sleep(x_train, n_fft = 100, hop_length=50, win_length=100, phase = False, stack_axes = True):
    
    train_tensor = x_train.squeeze(1).reshape(1, -1)
    
    x = torch.stft(
        input=train_tensor,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=torch.hann_window(win_length),
        center=False,
        return_complex=True)  # [num_channels, num_bins, num_frames]

    x_cartesian = src.utils.complex_to_cartesian(x)
    x_magnitude = src.utils.complex_to_magnitude(x, expand=True)

    x = x_cartesian if phase else x_magnitude
    if stack_axes:
        # Stack all spectrograms and put time dim first:
        # [num_channels, num_bins, num_frames, stft_parts] ->
        # [num_frames, num_channels x num_bins x stft_parts]
        x = einops.rearrange(x, 'C F T P -> T (C F P)')  # P=2
    else:
        x = einops.rearrange(x, 'C F T P -> T C F P')
    
    return x

def feature_kpi(x_train, n_fft = 10, hop_length=5, win_length=10, phase = False, stack_axes = True):
    
    
   
    train_tensor = torch.tensor(x_train).unsqueeze(1).reshape(1, -1)
    
    x = torch.stft(
        input=train_tensor,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=torch.hann_window(win_length),
        center=False,
        return_complex=True)  # [num_channels, num_bins, num_frames]

    x_cartesian = src.utils.complex_to_cartesian(x)
    x_magnitude = src.utils.complex_to_magnitude(x, expand=True)

    x = x_cartesian if phase else x_magnitude
    if stack_axes:
        # Stack all spectrograms and put time dim first:
        # [num_channels, num_bins, num_frames, stft_parts] ->
        # [num_frames, num_channels x num_bins x stft_parts]
        x = einops.rearrange(x, 'C F T P -> T (C F P)')  # P=2
    else:
        x = einops.rearrange(x, 'C F T P -> T C F P')
    
    return x

def windowed_labels(
    labels,
    num_labels,
    frame_length,
    frame_step=None,
    pad_end=False,
    kind='density',
):
    """Generates labels that correspond to STFTs

    With kind=None we are able to split the given labels
    array into batches. (T, C) -> (B, T', C)

    Parameters
    ----------
    labels : np.array

    Returns
    -------
    np.array
    """
    labels = torch.tensor(labels).view(-1)
    
    # Labels should be a single vector (int-likes) or kind has to be None
    labels = np.asarray(labels)
    
    if kind is not None and not labels.ndim == 1:
        raise ValueError('Labels must be a vector')
    if not (labels >= 0).all():
        raise ValueError('All labels must be >= 0')
    if not (labels < num_labels).all():
        raise ValueError(f'All labels must be < {num_labels} (num_labels)')
    # Kind determines how labels in each window should be processed
    if not kind in {'counts', 'density', 'onehot', 'argmax', None}:
        raise ValueError('`kind` must be in {counts, density, onehot, argmax, None}')
    # Let frame_step default to one full frame_length
    frame_step = frame_length if frame_step is None else frame_step
    # Process labels with a sliding window. TODO: vectorize?
    output = []
    for i in range(0, len(labels), frame_step):
        chunk = labels[i:i+frame_length]
        chunk = chunk.astype(int)
        # Ignore incomplete end chunk unless padding is enabled
        if len(chunk) < frame_length and not pad_end:
            continue
        # Just append the chunk if kind is None
        if kind == None:
            output.append(chunk)
            continue
        # Count the occurences of each label
        counts = np.bincount(chunk, minlength=num_labels)
        # Then process based on kind
        if kind == 'counts':
            output.append(counts)
        elif kind == 'density':
            output.append(counts / len(chunk))
        elif kind == 'onehot':
            one_hot = np.zeros(num_labels)
            one_hot[np.argmax(counts)] = 1
            output.append(one_hot)
        elif kind == 'argmax':
            output.append(np.argmax(counts))
    if pad_end:
        return output
    else:
        return torch.tensor(output)
    
class STFTDataset(Dataset):
    
    def __init__(self, data_path, class_to_exclude=3, n_fft = 250, hop_length=125, win_length=250, seq_length=500, num_labels=4):
        """
        Args:
            x_data (Tensor): The input features, e.g., from STFT.
            y_data (Tensor): The corresponding labels, windowed and processed.
            seq_length (int): The length of each sequence.
        """
        
        # Load each of the pickle files
        tensor_data = load_pickle_file(f'./{data_path}/tensor_data.pkl')
        tensor_label = load_pickle_file(f'./{data_path}/tensor_label.pkl')
        
        x_data = feature_stft(tensor_data, n_fft = n_fft, hop_length=hop_length, win_length=win_length)
        y_data = windowed_labels(labels=tensor_label, num_labels=num_labels, frame_length=n_fft, frame_step=hop_length, kind='argmax')

        self.x_data = x_data
        self.y_data = y_data
        self.seq_length = seq_length
        # self.class_to_exclude = class_to_exclude
        
        # Create a mask that filters out the class_to_exclude
        # mask = self.y_data != self.class_to_exclude

        # Apply the mask to filter the data
        # self.x_data = self.x_data[mask]
        # self.y_data = self.y_data[mask]
        
    def __len__(self):
        # Return the number of full sequences in the dataset
        return len(self.x_data) // self.seq_length

    def __getitem__(self, idx):
        """
        Returns a tuple (input, label) for the given index.
        The input is reshaped to (seq_length, features).
        """
        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length

        # Extract the sequence of data and corresponding labels
        x_seq = self.x_data[start_idx:end_idx]
        y_seq = self.y_data[start_idx:end_idx]

        return x_seq, y_seq

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
        final_indices = [idx for batch in final_indices for idx in batch]
        
        return iter(final_indices)

    def __len__(self):
        return len(self.data_source)
    

# Wrapper dataset class to flatten the batches
class FlattenedDataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        self.num_batches = len(original_dataset)
        self.batch_size = original_dataset[0][0].shape[0]  # Assuming shape [599, 156]

    def __len__(self):
        return self.num_batches * self.batch_size

    def __getitem__(self, idx):
        batch_idx = idx // self.batch_size
        sample_idx = idx % self.batch_size
        data_batch, label_batch = self.original_dataset[batch_idx]
        
        return data_batch[sample_idx], label_batch[sample_idx]
    
class SLEEPDataset(Dataset):
    
    def __init__(self, data_path, n_fft = 178, hop_length=89, win_length=178, seq_length=119, num_labels=5, eval=False):
        """
        Args:
            x_data (Tensor): The input features, e.g., from STFT.
            y_data (Tensor): The corresponding labels, windowed and processed.
            seq_length (int): The length of each sequence.
        """
        
        # Load each of the pickle files
        if eval:
            data = torch.load(f'{data_path}/val.pt')
        else:
            data = torch.load(f'{data_path}/train.pt')

        x_data = data['samples'].squeeze()
        self.y_data = data['labels']

        self.x_data = feature_sleep(x_data, n_fft = n_fft, hop_length=hop_length, win_length=win_length)
        self.seq_length = seq_length
        
    def __len__(self):
        # Return the number of full sequences in the dataset
        return len(self.x_data) // self.seq_length

    def __getitem__(self, idx):
        """
        Returns a tuple (input, label) for the given index.
        The input is reshaped to (seq_length, features).
        """
        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length

        # Extract the sequence of data and corresponding labels
        x_seq = self.x_data[start_idx:end_idx]
        y_seq = self.y_data[start_idx:end_idx]

        return x_seq, y_seq
    

class KpiDataset(Dataset):

    def __init__(self, data_path, n_fft = 250, hop_length=125, win_length=250, seq_length=500, num_labels=2):
        """
        Args:
            x_data (Tensor): The input features, e.g., from STFT.
            y_data (Tensor): The corresponding labels, windowed and processed.
            seq_length (int): The length of each sequence.
        """
        
        
        df = pd.read_csv(f'{data_path}/train.csv')
        tensor_data = df['value'].values 
        tensor_label = df['label'].values  
        
        x_data = feature_kpi(tensor_data, n_fft = n_fft, hop_length=hop_length, win_length=win_length)
        y_data = windowed_labels(labels=tensor_label, num_labels=num_labels, frame_length=n_fft, frame_step=hop_length, kind='argmax')

        self.x_data = x_data
        self.y_data = y_data
        self.seq_length = seq_length
        
    def __len__(self):
        # Return the number of full sequences in the dataset
        return len(self.x_data) // self.seq_length

    def __getitem__(self, idx):
        """
        Returns a tuple (input, label) for the given index.
        The input is reshaped to (seq_length, features).
        """
        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length

        # Extract the sequence of data and corresponding labels
        x_seq = self.x_data[start_idx:end_idx]
        y_seq = self.y_data[start_idx:end_idx]

        return x_seq, y_seq