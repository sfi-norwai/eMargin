
import torch
from torch.utils.data import Subset
import random
from src.loader.dataloader import FlattenedDataset
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score




def load_balanced_dataset(dataset, class_counts):
    # Initialize dictionary to store indices of each class
    class_indices = {label: [] for label in class_counts.keys()}
    
    # Populate class_indices with indices of each class
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        label = label.item()  # Ensure the label is a scalar
        if label in class_indices:
            class_indices[label].append(idx)

    # Ensure each class has the required number of instances
    balanced_indices = []
    for label, count in class_counts.items():
        if len(class_indices[label]) >= count:
            balanced_indices.extend(random.sample(class_indices[label], count))
        else:
            raise ValueError(f"Not enough instances of class {label} to satisfy the requested count")

    # Create a subset of the dataset with the balanced indices
    balanced_subset = Subset(dataset, balanced_indices)
    return balanced_subset

def clustering_evaluation(model, valid_dataset, config):


    desired_count_per_class = config.class_count
    class_dict = config.class_dict

    flattened_data = FlattenedDataset(valid_dataset)

    # Load balanced dataset

    balanced_dataset = load_balanced_dataset(flattened_data, desired_count_per_class)
    valid_balanced_dataloader = DataLoader(balanced_dataset, batch_size=config.display_batch, shuffle=False, num_workers=config.NUM_WORKERS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                  
    for batch in valid_balanced_dataloader:
        images, real_label = batch
        images = images.view(-1, 1, images.shape[-1]).to(device)

    
    time_features = model.encode(images)

    try:
        kmeans = KMeans(n_clusters=len(class_dict), random_state=1, n_init=10).fit(time_features.cpu().detach().squeeze())
        labeli = kmeans.labels_
        # Calculate Davies-Bouldin Index
        db_index2 = davies_bouldin_score(time_features.cpu().detach().squeeze(), labeli)
        ch_index2 = calinski_harabasz_score(time_features.cpu().detach().squeeze(), labeli)
        slh_index2 = silhouette_score(time_features.cpu().detach().squeeze(), labeli)
        print(f"DB Index: {db_index2:.2f}, CH Index: {ch_index2:.2f}, SLH Index: {slh_index2:.2f}")
        # nmi_index2 = normalized_mutual_info_score(real_label, labeli)
        # ari_index2 = adjusted_rand_score(real_label, labeli)

    except:
        db_index2 = 0
        ch_index2 = 0
        slh_index2 = 0
        print(f"DB Index: {db_index2:.2f}, CH Index: {ch_index2:.2f}, SLH Index: {slh_index2:.2f}")

    
    return {
            'Davies-Bouldin Index Features': db_index2,
            'Calinski Harabasz Index Features': ch_index2,
            'Silhouette Index Features': slh_index2
        }

    # return {
    #             'Normalized-Mutual Info': nmi_index2,
    #             'Adjusted Rand Index': ari_index2
    #         }