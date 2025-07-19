import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset


class InputEmbeddingPosEncoding(nn.Module):
    def __init__(self, feature_dim=512):
        super(InputEmbeddingPosEncoding, self).__init__()
        self.lin_proj_layer = nn.Linear(in_features=156, out_features=1500)
        self.lin_proj_layer1 = nn.Linear(in_features=1500, out_features=feature_dim)
        # self.lin_proj_layer2 = nn.Linear(in_features=500, out_features=256)
        self.pos_encoder = AbsolutePositionalEncoding()

    def forward(self, x):
        x = self.lin_proj_layer(x)
        x = self.lin_proj_layer1(x)
        # x = self.lin_proj_layer2(x)
        x = self.pos_encoder(x)
        return x

class AbsolutePositionalEncoding(nn.Module):
    def __init__(self):
        super(AbsolutePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.0)

    def forward(self, x):
        return self.dropout(x)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, feature_dim=512):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(feature_dim, num_heads=10)
        self.linear1 = nn.Linear(feature_dim, 2048)
        self.dropout1 = nn.Dropout(p=0.1)
        self.linear2 = nn.Linear(2048, feature_dim)
        self.dropout2 = nn.Dropout(p=0.1)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x, _ = self.self_attn(x, x, x)
        x = x.permute(1, 0, 2)
        residual = x
        x = self.norm1(x)
        x = F.relu(self.linear1(x))
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        x += residual
        x = self.norm2(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, feature_dim=512):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(feature_dim) for _ in range(4)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TransformerEncoderNetwork(nn.Module):
    def __init__(self, feature_dim=512):
        super(TransformerEncoderNetwork, self).__init__()
        self.emb = InputEmbeddingPosEncoding(feature_dim)
        self.transformer_encoder = TransformerEncoder(feature_dim)
        
    def forward(self, x):
        x = self.emb(x)
        x = self.transformer_encoder(x)
        
        return x
    
class FeatureProjector(nn.Module):
    def __init__(self, input_size=156, output_size=32):
        super(FeatureProjector, self).__init__()
        
        # 1D Convolutional Layers
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=128, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=output_size, kernel_size=1)
        
        # Batch Normalization
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(output_size)
        
    def forward(self, x):
        # Input x shape: (batch_size, sequence_length, input_size)
        x = x.float()
        
        # Permute to match Conv1D input: (batch_size, input_size, sequence_length)
        x = x.permute(0, 2, 1)
        
        # First convolutional layer
        x = self.conv1(x)  # Shape: (batch_size, 128, sequence_length)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Second convolutional layer
        x = self.conv2(x)  # Shape: (batch_size, 64, sequence_length)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Third convolutional layer
        x = self.conv3(x)  # Shape: (batch_size, output_size, sequence_length)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Permute back to original order: (batch_size, sequence_length, output_size)
        x = x.permute(0, 2, 1)
        
        return x

class SLEEPDataset(Dataset):
    
    def __init__(self, data_path, seq_length=500):
        """
        Args:
            x_data (Tensor): The input features, e.g., from STFT.
            y_data (Tensor): The corresponding labels, windowed and processed.
            seq_length (int): The length of each sequence.
        """
        
        # Load each of the pickle files
        data = torch.load(f'./{data_path}/train.pt')
        self.x_data = data['samples'].squeeze()
        self.y_data = data['labels']
    
        
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
       
class Discriminator(torch.nn.Module):
    def __init__(self, feature_dim,  device):
        super(Discriminator, self).__init__()
        self.device = device
        self.feature_dim = feature_dim

        self.model = torch.nn.Sequential(torch.nn.Linear(2*self.feature_dim, 4*self.feature_dim),
                                         torch.nn.ReLU(inplace=True),
                                         torch.nn.Dropout(0.5),
                                         torch.nn.Linear(4*self.feature_dim, 1))

        torch.nn.init.xavier_uniform_(self.model[0].weight)
        torch.nn.init.xavier_uniform_(self.model[3].weight)

    def forward(self, x, x_tild):
        """
        Predict the probability of the two inputs belonging to the same neighbourhood.
        """
        x_all = torch.cat([x, x_tild], -1)
        p = self.model(x_all)
        return p.view((-1,))