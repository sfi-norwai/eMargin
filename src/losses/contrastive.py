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
import numpy
import random


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def hierarchical_contrastive_loss(z1, z2, alpha=0.5, temporal_unit=0):
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        print(z1.shape)
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        if d >= temporal_unit:
            if 1 - alpha != 0:
                loss += (1 - alpha) * temporal_contrastive_loss(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        d += 1
    return loss / d

def instance_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss

def temporal_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    return loss

class HATCL_LOSS(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(HATCL_LOSS, self).__init__()
        self.temperature = temperature

    def forward(self, features):
        # Normalize the feature vectors
        features_normalized = F.normalize(features, dim=-1, p=2)

        # Calculate the cosine similarity matrix
        similarities = torch.matmul(features_normalized, features_normalized.T)
        
        exp_similarities = torch.exp(similarities / self.temperature)
        
        # Removing the similarity of a window with itself i.e main diagonal
        exp_similarities = exp_similarities - torch.diag(exp_similarities.diag())        

        # Lower diagonal elements represent positive pairs
        positives = torch.diagonal(exp_similarities, offset=-1)

        # The denominator is the sum of the column vectors minus the positives
        denominator = torch.sum(exp_similarities[:,:-1], dim=0) - positives
        
        # Calculate NT-Xent loss
        loss = -torch.log(positives / denominator).mean()

        return loss


class LS_HATCL_LOSS(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(LS_HATCL_LOSS, self).__init__()
        self.temperature = temperature

    def forward(self, features):
        
        # Normalize the feature vectors
        features_normalized = torch.nn.functional.normalize(features, p=2, dim=-1)

        # Calculate the cosine similarity matrix
        similarities = torch.matmul(features_normalized, features_normalized.T)

        
        exp_similarities = torch.exp(similarities / self.temperature)
        
        # Removing the similarity of a window with itself i.e main diagonal
        exp_similarities = exp_similarities - torch.diag(exp_similarities.diag())        

        # Lower diagonal elements represent positive pairs
        lower_diag = torch.diagonal(exp_similarities, offset=-1)
        
        # The numerator is the sum of shifted left and right of the positive pairs
        numerator = lower_diag[1:] + lower_diag[:-1]
        
        # The denominator is the sum of the column vectors minus the positives
        denominator = torch.sum(exp_similarities[:,:-2], dim=0) - lower_diag[:-1]\
                + (torch.sum(exp_similarities[:,1:-1], dim=0)  - (lower_diag[1:] + lower_diag[:-1]))
        
        
        # Calculate NT-Xent loss
        loss = -torch.log(numerator / denominator).mean()
        
#         print("Similarities: ", similarities)
#         print("Exp Similarities: ", exp_similarities)
#         print("Numerator: ", numerator)
#         print("Denominator: ", denominator)
        
        return loss

class RAND_HATCL_LOSS(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(RAND_HATCL_LOSS, self).__init__()
        self.temperature = temperature

    def forward(self, features):

        # Shuffle along the NT dimension
        indices = torch.randperm(features.size(0))  # features.size(0) == NT
        featuresR = features[indices]

        # Normalize the feature vectors
        features_normalized = torch.nn.functional.normalize(features, p=2, dim=-1)
        features_normalizedR = torch.nn.functional.normalize(featuresR, p=2, dim=-1)


        # Calculate the cosine similarity matrix
        similarities = torch.matmul(features_normalized, features_normalized.T)

        # Calculate the cosine similarity matrix
        similaritiesR = torch.matmul(features_normalizedR, features_normalizedR.T)

        
        exp_similarities = torch.exp(similarities / self.temperature)
        exp_similaritiesR = torch.exp(similaritiesR / self.temperature)
        
        # Removing the similarity of a window with itself i.e main diagonal
        exp_similarities = exp_similarities - torch.diag(exp_similarities.diag())
        exp_similaritiesR = exp_similaritiesR - torch.diag(exp_similaritiesR.diag())       


        # Lower diagonal elements represent positive pairs
        lower_diag = torch.diagonal(exp_similarities, offset=-1)
        
        
        # The numerator is the sum of shifted left and right of the positive pairs
        numerator = lower_diag[1:] + lower_diag[:-1]
        
        # The denominator is the sum of the column vectors minus the positives
        denominator = torch.sum(exp_similaritiesR[:,:-2], dim=0) - lower_diag[:-1]\
                + (torch.sum(exp_similaritiesR[:,1:-1], dim=0)  - (lower_diag[1:] + lower_diag[:-1]))
        
        
        # Calculate NT-Xent loss
        loss = -torch.log(numerator / denominator).mean()

        return loss
    
class NN_HATCL_LOSS(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(NN_HATCL_LOSS, self).__init__()
        self.temperature = temperature

    def forward(self, features):
        # Normalize the feature vectors
        features_normalized = F.normalize(features, dim=-1, p=2)

        # Calculate the cosine similarity matrix
        similarities = torch.matmul(features_normalized, features_normalized.T)
        
        exp_similarities = torch.exp(similarities / self.temperature)
        
        # Removing the similarity of a window with itself i.e main diagonal
        exp_similarities = exp_similarities - torch.diag(exp_similarities.diag())        

        # Lower diagonal elements represent positive pairs
        positives = torch.diagonal(exp_similarities, offset=-1)
        
        
#         # Normalize the feature vectors
#         features_normalized2 = F.normalize(features2, dim=-1, p=2)

#         # Calculate the cosine similarity matrix
#         similarities2 = torch.matmul(features_normalized2, features_normalized2.T)
        
#         exp_similarities2 = torch.exp(similarities2 / self.temperature)
        
#         # Removing the similarity of a window with itself i.e main diagonal
#         exp_similarities2 = exp_similarities2 - torch.diag(exp_similarities2.diag())        

#         # Lower diagonal elements represent positive pairs
#         positives2 = torch.diagonal(exp_similarities2, offset=-1)
        
#         # The denominator is the sum of the column vectors minus the positives
#         denominator = torch.sum(exp_similarities[:,:-1], dim=0) - positives2
        
        
        
        
        # Calculate NT-Xent loss
        loss = -torch.log(positives).mean()

        return loss

class RAN_HATCL_LOSS(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(RAN_HATCL_LOSS, self).__init__()
        self.temperature = temperature

    def forward(self, features, features2):
        # Normalize the feature vectors
        features_normalized = F.normalize(features, dim=-1, p=2)

        # Calculate the cosine similarity matrix
        similarities = torch.matmul(features_normalized, features_normalized.T)
        
        exp_similarities = torch.exp(similarities / self.temperature)
        
        # Removing the similarity of a window with itself i.e main diagonal
        exp_similarities = exp_similarities - torch.diag(exp_similarities.diag())        

        # Lower diagonal elements represent positive pairs
        positives = torch.diagonal(exp_similarities, offset=-1)
        
        
        # Normalize the feature vectors
        features_normalized2 = F.normalize(features2, dim=-1, p=2)

        # Calculate the cosine similarity matrix
        similarities2 = torch.matmul(features_normalized2, features_normalized2.T)
        
        exp_similarities2 = torch.exp(similarities2 / self.temperature)
        
        # Removing the similarity of a window with itself i.e main diagonal
        exp_similarities2 = exp_similarities2 - torch.diag(exp_similarities2.diag())        

        # Lower diagonal elements represent positive pairs
        positives2 = torch.diagonal(exp_similarities2, offset=-1)
        
        # The denominator is the sum of the column vectors minus the positives
        denominator = torch.sum(exp_similarities2[:,:-1], dim=0) - positives2
        
        
        
        
        # Calculate NT-Xent loss
        loss = -torch.log(positives/denominator).mean()

        return loss


# Define custom dataset class
class AugmentedImageDataset(Dataset):
    def __init__(self, original_dataset, transform):
        self.original_dataset = original_dataset
        self.transform = transform

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        # Get original image and label from the original dataset
        original_image, label = self.original_dataset[idx]

        # Apply transformations to the original image to get augmented image
        augmented_image = self.transform(original_image)

        return original_image, augmented_image, label

# Define transformations for augmentation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(size=64),
    transforms.RandomGrayscale(p=0.2),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor()
])


class SPAT_HATCL_LOSS(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(SPAT_HATCL_LOSS, self).__init__()
        self.temperature = temperature

    def forward(self, features, features2):
        # Normalize the feature vectors
        features_normalized = F.normalize(features, dim=-1, p=2)
        features_normalized2 = F.normalize(features2, dim=-1, p=2)
        

        # Calculate the cosine similarity matrix
        similarities = torch.matmul(features_normalized, features_normalized2.T)
        
        
        exp_similarities = torch.exp(similarities / self.temperature)
        
        
        # main diagonal elements represent positive pairs
        numerator = torch.diag(exp_similarities.diag())
        
        
        # Removing the similarity of a window with its augmentation i.e main diagonal
        exp_similarities = exp_similarities - numerator
        

        # The denominator is the sum of the column vectors minus the positives
        numerator = torch.sum(numerator, dim=0)
        
        # The denominator is the sum of the column vectors minus the positives
        denominator = torch.sum(exp_similarities, dim=0)
        
        # Calculate NT-Xent loss
        loss = -torch.log(numerator/denominator).mean()

        return loss
    
class MARGIN_LOSS(torch.nn.Module):
    def __init__(self, temperature=0.5, margin=5):
        super(MARGIN_LOSS, self).__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(self, features, sim_vector):
        # Normalize the feature vectors
        features_normalized = F.normalize(features, dim=-1, p=2)

        # Calculate the cosine similarity matrix
        similarities = torch.matmul(features_normalized, features_normalized.T)

        
        


        # Removing the similarity of a window with itself i.e main diagonal
        similarities = similarities - torch.diag(similarities.diag())  
    
        # Lower diagonal elements represent positive pairs
        positives = torch.diagonal(similarities, offset=-1)
        

        # The denominator is the sum of the column vectors minus the positives
        denominator = similarities[:,:-1]
        

        # The denominator is the sum of the column vectors minus the positives
        new_denominator = (1-sim_vector)*0.5*( denominator )**2 + \
                      sim_vector*0.5*torch.max(torch.tensor(0), (self.margin - (denominator)))**2
        
        exp_numerator = torch.exp(positives / self.temperature)
        

        exp_denominator = torch.exp(new_denominator)


        exp_denominator = torch.sum(exp_denominator, dim=0) - exp_numerator
        
        # Calculate NT-Xent loss
        loss = -torch.log(exp_numerator / exp_denominator).mean()
        
        return loss
    
class LS_MARGIN_LOSS(torch.nn.Module):
    def __init__(self, temperature=0.5, margin=5):
        super(LS_MARGIN_LOSS, self).__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(self, features, sim_vector):
        # Normalize the feature vectors
        features_normalized = F.normalize(features, dim=-1, p=2)

        # Calculate the cosine similarity matrix
        similarities = torch.matmul(features_normalized, features_normalized.T)

        # Lower diagonal elements represent positive pairs
        lower_diag = torch.diagonal(similarities, offset=-1)
        exp_numerator = torch.exp(lower_diag[1:] / self.temperature) + torch.exp(lower_diag[:-1] / self.temperature)
    
    
        # The denominator is the sum of the column vectors minus the positives
        new_similarities = -(1-sim_vector)*0.5*( similarities )**2 + \
                    sim_vector*0.5*torch.max(torch.tensor(0), (self.margin - (similarities)))**2    

        # Remove negative and introduced gamma for double margin to avoid NaN values
        exp_sim = torch.exp(0.05*new_similarities  / self.temperature)
    
        exp_similarities = exp_sim - torch.diag(exp_sim.diag())
    
        # The denominator is the sum of the column vectors minus the positives
        exp_denominator = torch.sum(exp_similarities[:,:-2], dim=0) - torch.exp(lower_diag[:-1] / self.temperature)\
                + (torch.sum(exp_similarities[:,1:-1], dim=0)  - (exp_numerator))
    
        # Calculate NT-Xent loss
        loss = -torch.log(exp_numerator / (exp_denominator + exp_numerator)).mean()
    
        return loss
    
class TripletLoss(torch.nn.modules.loss._Loss):
    
    def __init__(self, compared_length, nb_random_samples, negative_penalty, output_size):
        super(TripletLoss, self).__init__()
        self.compared_length = compared_length
        if self.compared_length is None:
            self.compared_length = numpy.inf
        self.nb_random_samples = nb_random_samples
        self.negative_penalty = negative_penalty
        self.output_size = output_size

    def forward(self, batch, encoder, train, save_memory=False):
        batch_size = batch.size(0)
        train_size = train.size(0)
        length = min(self.compared_length, train.size(2))
        
#         print("length: ", length)

        # For each batch element, we pick nb_random_samples possible random
        # time series in the training set (choice of batches from where the
        # negative examples will be sampled)
        samples = numpy.random.choice(
            train_size, size=(self.nb_random_samples, batch_size)
        )
        samples = torch.LongTensor(samples)
        
#         print("samples:", samples)

        # Choice of length of positive and negative samples
        length_pos_neg = numpy.random.randint(1, high=length + 1)
#         print("length_pos_neg: ", length_pos_neg)

        # We choose for each batch example a random interval in the time
        # series, which is the 'anchor'
        random_length = numpy.random.randint(
            length_pos_neg, high=length + 1
        )  # Length of anchors
#         print("random_length: ", random_length)
        
        length_pos_neg = random_length
        
        beginning_batches = numpy.random.randint(
            0, high=length - random_length + 1, size=batch_size
        )  # Start of anchors
#         print("beginning_batches: ", beginning_batches)
        

        # The positive samples are chosen at random in the chosen anchors
        beginning_samples_pos = numpy.random.randint(
            0, high=random_length - length_pos_neg + 1, size=batch_size
        )  
#         print("beginning_samples_pos: ", beginning_samples_pos)
        
        # Start of positive samples in the anchors
        # Start of positive samples in the batch examples
        beginning_positive = beginning_batches + beginning_samples_pos
#         print("beginning_positive: ", beginning_positive)
        
        # End of positive samples in the batch examples
        end_positive = beginning_positive + length_pos_neg
#         print("end_positive: ", end_positive)
        
        

        # We randomly choose nb_random_samples potential negative samples for
        # each batch example
        beginning_samples_neg = numpy.random.randint(
            0, high=length - length_pos_neg + 1,
            size=(self.nb_random_samples, batch_size)
        )
        
        default_rep = torch.cat(
            [batch[
                j: j + 1, :,
                beginning_batches[j]: beginning_batches[j] + random_length
            ] for j in range(batch_size)]
        )
        
        default_rep_transposed = default_rep.transpose(1, 2)

        representation = encoder(default_rep_transposed)  # Anchors representations
        
        positive_rep = torch.cat(
            [batch[
                j: j + 1, :, end_positive[j] - length_pos_neg: end_positive[j]
            ] for j in range(batch_size)]
        )
        positive_rep_transposed = positive_rep.transpose(1, 2)
        positive_representation = encoder(positive_rep_transposed)  # Positive samples representations

        size_representation = representation.size(1)
        size_posrepresentation = positive_representation.size(1)
        # Positive loss: -logsigmoid of dot product between anchor and positive
        # representations
        
        
#         print(representation.shape)
#         print(positive_representation.shape)
        
        
        
        loss = -torch.mean(torch.nn.functional.logsigmoid(torch.bmm(
            representation.reshape(batch_size, self.output_size, size_representation),
            positive_representation.reshape(batch_size, size_posrepresentation, self.output_size)
        )))

        # If required, backward through the first computed term of the loss and
        # free from the graph everything related to the positive sample
        if save_memory:
            loss.backward(retain_graph=True)
            loss = 0
            del positive_representation
            torch.cuda.empty_cache()

        multiplicative_ratio = self.negative_penalty / self.nb_random_samples
        for i in range(self.nb_random_samples):
            # Negative loss: -logsigmoid of minus the dot product between
            # anchor and negative representations
            
            negative_rep = torch.cat([train[samples[i, j]: samples[i, j] + 1][
                    :, :,
                    beginning_samples_neg[i, j]:
                    beginning_samples_neg[i, j] + length_pos_neg
                ] for j in range(batch_size)])
            
            negative_rep_transposed = negative_rep.transpose(1, 2)
            negative_representation = encoder(negative_rep_transposed)
            
#             print(negative_representation.shape)
            
            loss += multiplicative_ratio * -torch.mean(
                torch.nn.functional.logsigmoid(-torch.bmm(
                    representation.reshape(batch_size, self.output_size, size_representation),
                    negative_representation.reshape(
                        batch_size, size_representation, self.output_size
                    )
                ))
            )
            # If required, backward through the first computed term of the loss
            # and free from the graph everything related to the negative sample
            # Leaves the last backward pass to the training procedure
            if save_memory and i != self.nb_random_samples - 1:
                loss.backward(retain_graph=True)
                loss = 0
                del negative_representation
                torch.cuda.empty_cache()

        return loss

def local_infoNCE(z1, z2, pooling='max',temperature=1.0, k = 16):
    #   z1, z2    B X T X D
    B = z1.size(0)
    T = z1.size(1)
    D = z1.size(2)
    crop_size = int(T/k)
    crop_leng = crop_size*k

    # random start?
    start = random.randint(0,T-crop_leng)
    crop_z1 = z1[:,start:start+crop_leng,:]
    crop_z1 = crop_z1.view(B ,k,crop_size,D)


    # crop_z2 = z2[:,start:start+crop_leng,:]
    # crop_z2 = crop_z2.view(B ,k,crop_size,D)


    if pooling=='max':
        crop_z1 = crop_z1.reshape(B*k,crop_size,D)
        crop_z1_pooling = F.max_pool1d(crop_z1.transpose(1, 2).contiguous(), kernel_size=crop_size).transpose(1, 2).reshape(B,k,D)

        # crop_z2 = crop_z2.reshape(B*k,crop_size,D)
        # crop_z2_pooling = F.max_pool1d(crop_z2.transpose(1, 2).contiguous(), kernel_size=crop_size).transpose(1, 2)

    elif pooling=='mean':
        crop_z1_pooling = torch.unsqueeze(torch.mean(z1,1),1)
        # crop_z2_pooling = torch.unsqueeze(torch.mean(z2,1),1)

    crop_z1_pooling_T = crop_z1_pooling.transpose(1,2)

    # B X K * K
    similarity_matrices = torch.bmm(crop_z1_pooling, crop_z1_pooling_T)

    labels = torch.eye(k-1, dtype=torch.float32)
    labels = torch.cat([labels,torch.zeros(1,k-1)],0)
    labels = torch.cat([torch.zeros(k,1),labels],-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pos_labels = labels.to(device)
    pos_labels[k-1,k-2]=1.0


    neg_labels = labels.T + labels + torch.eye(k)
    neg_labels[0,2]=1.0
    neg_labels[-1,-3]=1.0
    neg_labels = neg_labels.to(device)


    similarity_matrix = similarity_matrices[0]

    # select and combine multiple positives
    positives = similarity_matrix[pos_labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~neg_labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)

    logits = logits / temperature
    logits = -F.log_softmax(logits, dim=-1)
    loss = logits[:,0].mean()

    return loss



def global_infoNCE(z1, z2, pooling='max',temperature=1.0):
    if pooling == 'max':
        z1 = F.max_pool1d(z1.transpose(1, 2).contiguous(), kernel_size=z1.size(1)).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2).contiguous(), kernel_size=z2.size(1)).transpose(1, 2)
    elif pooling == 'mean':
        z1 = torch.unsqueeze(torch.mean(z1, 1), 1)
        z2 = torch.unsqueeze(torch.mean(z2, 1), 1)

    # return instance_contrastive_loss(z1, z2)
    return InfoNCE(z1,z2,temperature)

def InfoNCE(z1, z2, temperature=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = z1.size(0)

    features = torch.cat([z1, z2], dim=0).squeeze(1)  # 2B x T x C

    labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    # features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
    logits = logits / temperature
    logits = -F.log_softmax(logits, dim=-1)
    loss = logits[:,0].mean()

    return loss


class SimMTMLoss(nn.Module):

    def __init__(self, device, temperature):
        super(SimMTMLoss, self).__init__()
        self.device = device
        self.temperature = temperature
        
        self.bce = torch.nn.BCELoss()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)
        
        self.kl = torch.nn.KLDivLoss(reduction='batchmean')
        
    def get_positive_and_negative_mask(self, similarity_matrix, cur_batch_size, oral_batch_size):
        
        diag = np.eye(cur_batch_size)
        mask = torch.from_numpy(diag)
        mask = mask.type(torch.bool)
        
        positives_mask = np.zeros(similarity_matrix.size())
        for i in range(cur_batch_size//oral_batch_size):
            ll = np.eye(cur_batch_size, cur_batch_size, k=oral_batch_size*i)
            lr = np.eye(cur_batch_size, cur_batch_size, k=-oral_batch_size*i)
            positives_mask += ll
            positives_mask += lr
        
        positives_mask = torch.from_numpy(positives_mask)
        positives_mask[mask] = 0

        negatives_mask = 1 - positives_mask
        negatives_mask[mask] = 0
        
        return positives_mask.type(torch.bool), negatives_mask.type(torch.bool)

    def forward(self, batch_emb_om, batch_x):
        
        cur_batch_shape = batch_emb_om.shape
        oral_batch_shape = batch_x.shape
        
        # get similarity matrix among mask samples
        norm_emb = F.normalize(batch_emb_om, dim=1)
        similarity_matrix = torch.matmul(norm_emb, norm_emb.transpose(0, 1))
        
        # get positives and negatives similarity
        positives_mask, negatives_mask = self.get_positive_and_negative_mask(similarity_matrix, cur_batch_shape[0], oral_batch_shape[0])

        positives = similarity_matrix[positives_mask].view(cur_batch_shape[0], -1)
        negatives = similarity_matrix[negatives_mask].view(cur_batch_shape[0], -1)
        
        # generate predict and target probability distributions matrix
        logits = torch.cat((positives, negatives), dim=-1) 
        y_true = torch.cat((torch.ones(cur_batch_shape[0], positives.shape[-1]) / positives.shape[-1],  torch.zeros(cur_batch_shape[0], negatives.shape[-1])), dim=-1).to(self.device).float()
        
        # multiple positives - KL divergence
        predict = self.log_softmax(logits / self.temperature)
        loss = self.kl(predict, y_true)
        
        return loss, similarity_matrix, logits


