import torch
from utils import take_per_row
from tqdm import tqdm
import torch
from models.contrastive import LS_HATCL_LOSS, HATCL_LOSS
from pytorch_lightning.loggers import WandbLogger
import wandb
from models.soft_losses import *
from tslearn.metrics import dtw, dtw_path,gak
from sklearn.preprocessing import MinMaxScaler
from src.models.ts2vecencoder import *
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


def get_COS(MTS_tr):
    MTS_tr = MTS_tr.view(MTS_tr.shape[0], -1)
    cos_sim_matrix = -cosine_similarity(MTS_tr)
    return cos_sim_matrix

def get_MDTW(MTS_tr):
    N = MTS_tr.shape[0]
    dist_mat = np.zeros((N,N))
    for i in tqdm(range(N)):
        for j in range(N):
            if i>j:
                mdtw_dist = dtw(MTS_tr[i], MTS_tr[j])
                dist_mat[i,j] = mdtw_dist
                dist_mat[j,i] = mdtw_dist
            elif i==j:
                dist_mat[i,j] = 0
            else :
                pass
    return dist_mat

def save_sim_mat(X_tr, min_ = 0, max_ = 1, multivariate=True, type_='DTW'):
    N = len(X_tr)
    if multivariate:
        assert type_=='DTW'
        dist_mat = get_COS(X_tr)
        
    # (1) distance matrix
    diag_indices = np.diag_indices(N)
    mask = np.ones(dist_mat.shape, dtype=bool)
    mask[diag_indices] = False
    temp = dist_mat[mask].reshape(N, N-1)
    dist_mat[diag_indices] = temp.min()
    
    # (2) normalized distance matrix
    scaler = MinMaxScaler(feature_range=(min_, max_))
    dist_mat = scaler.fit_transform(dist_mat)
    
    # (3) normalized similarity matrix
    return 1 - dist_mat

class Soft:
    '''The Soft model'''
    
    def __init__(
        self,
        args,
        config,
        device='cuda',
    ):
        '''
          Initialize a Soft model.

        '''
        
        self.args = args
        self.config = config
        super().__init__()
        
        self.device = device
        self.temporal_unit = 0
        self.max_train_length = 500
        self.net = TSEncoder(input_dims=args['feature_dim'], output_dims=args['out_features']).to(self.device)
        self.n_iters = 0
       
    
    def fit(self, train_dataset, ds_name, verbose=False):
        ''' Training the Soft model.
        
        Args:
            train_data (numpy.ndarray): The training data. It should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            n_epochs (Union[int, NoneType]): The number of epochs. When this reaches, the training stops.
            n_iters (Union[int, NoneType]): The number of iterations. When this reaches, the training stops. If both n_epochs and n_iters are not specified, a default setting would be used that sets n_iters to 200 for a dataset with size <= 100000, 600 otherwise.
            verbose (bool): Whether to print the training loss after each epoch.
            
        Returns:
            loss_log: a list containing the training losses on each epoch.
        '''
        
        train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size= self.args['batch_size'],
                shuffle = True,
                num_workers=self.config.NUM_WORKERS,
                drop_last = True,
            )
        
        # Wandb setup
        if self.config.WANDB:    
            proj_name = 'Dynamic_CL' + ds_name + str(self.config.SEED)
            run_name = 'Soft'

            wandb_logger = WandbLogger(project=proj_name)
            
            # Initialize Wandb
            wandb.init(project=proj_name, name=run_name)
            wandb.watch(self.net, log='all', log_freq=100)

            # Update Wandb config
        
            wandb.config.update(self.args)
            wandb.config.update({
                'Algorithm': f'{run_name}',
                'Dataset': f'{ds_name}',
                'Train_DS_size': len(train_dataset),
                'Batch_Size': self.args["batch_size"],
                'Epochs': self.args["epochs"],
                'Patience': self.config.PATIENCE,
                'Seed': self.config.SEED

            })
            wandb.run.name = run_name
            wandb.run.save()
        
        # Define loss function and optimizer

        if self.args['loss'] == 'HATCL_LOSS':
            cl_loss = HATCL_LOSS(temperature=self.args['temperature'])

        elif self.args['loss'] == 'LS_HATCL_LOSS':
            cl_loss = LS_HATCL_LOSS(temperature=self.args['temperature'])

        else:
            raise ValueError(f"Unsupported loss function: {self.args['loss']}")

        self.args['lr'] = float(self.args['lr'])

        if self.args['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args['lr'])  # Example optimizer

        elif self.args['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(self.net.parameters(), lr=self.args['lr'])  # Example optimizer

        elif self.args['optimizer'] == 'AdamW':
            optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.args['lr'])  # Example optimizer

        elif self.args['optimizer'] == 'Adadelta':
            optimizer = torch.optim.Adadelta(self.net.parameters(), lr=self.args['lr'])  # Example optimizer

        else:
            optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args['lr'])  # Example optimizer

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=self.config.PATIENCE)
        
        
        lambda_ = 0.5
        tau_temp = 2
        temporal_unit = 0
        soft_instance = False
        soft_temporal = False
        

        n_iters = self.args['iterations']
        pbar = tqdm(total=n_iters, desc="Training")
        epoch = 0

        while True:

            # Training phase
            self.net.train()  # Set the model to training mode
            train_running_loss = 0.0
            n_epoch_iters = 0

            for x, _ in train_loader:

                interrupted = False
                if n_iters is not None and self.n_iters >= n_iters:
                    interrupted = True
                    break

                soft_labels_batch = save_sim_mat(x)
                x = x.to(self.device)
               
                if self.max_train_length is not None and x.size(1) > self.max_train_length:
                    window_offset = np.random.randint(x.size(1) - self.max_train_length + 1)
                    x = x[:, window_offset : window_offset + self.max_train_length]
                x = x.to(self.device)
                
                ts_l = x.size(1)
                crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l+1)
                crop_left = np.random.randint(ts_l - crop_l + 1)
                crop_right = crop_left + crop_l
                crop_eleft = np.random.randint(crop_left + 1)
                crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
                crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))
                
                optimizer.zero_grad()
                
                out1 = self.net(take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft))
                out1 = out1[:, -crop_l:]
                
                out2 = self.net(take_per_row(x, crop_offset + crop_left, crop_eright - crop_left))
                out2 = out2[:, :crop_l]
                
                loss = hier_CL_soft(
                    out1,
                    out2,
                    soft_labels_batch,
                    lambda_= lambda_,
                    tau_temp = tau_temp,
                    temporal_unit = temporal_unit,
                    soft_temporal = soft_temporal, 
                    soft_instance = soft_instance
                )
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                    
                # Update training statistics
                n_epoch_iters += 1
                self.n_iters += 1
                pbar.update(1)

                train_running_loss += loss.item()

            if interrupted:
                break
            train_running_loss /= n_epoch_iters 

            
            epoch += 1
    
            if verbose:
                print(f"Epoch {epoch}, Train Loss: {train_running_loss:.4f}")

            # Log training loss to Wandb
            if self.config.WANDB:
                wandb.log({'Train Loss': train_running_loss, 'Epoch': epoch})
        try:   
            return train_running_loss
        except:
            return 0

    def encode(self, x, mask=None):
        self.net.eval()
        out = self.net(x.to(self.device, non_blocking=True), mask)

        return out


    def save(self, fn):
        ''' Save the model to a file.
        
        Args:
            fn (str): filename.
        '''
        torch.save(self.net.state_dict(), fn)
    
    def load(self, fn):
        ''' Load the model from a file.
        
        Args:
            fn (str): filename.
        '''
        state_dict = torch.load(fn, map_location=self.device)
        self.net.load_state_dict(state_dict)
