import torch
from models.contrastive import LS_HATCL_LOSS, HATCL_LOSS
from tqdm import tqdm
from src.models.attention_model import *
from pytorch_lightning.loggers import WandbLogger
import wandb
from models import TSEncoder
from statsmodels.tsa.stattools import adfuller
import numpy as np
import math

class Discriminator(torch.nn.Module):
    def __init__(self, input_size, device):
        super(Discriminator, self).__init__()
        self.device = device
        self.input_size = input_size

        self.model = torch.nn.Sequential(torch.nn.Linear(2*self.input_size, 4*self.input_size),
                                         torch.nn.ReLU(inplace=True),
                                         torch.nn.Dropout(0.5),
                                         torch.nn.Linear(4*self.input_size, 1))

        torch.nn.init.xavier_uniform_(self.model[0].weight)
        torch.nn.init.xavier_uniform_(self.model[3].weight)

    def forward(self, x, x_tild):
        """
        Predict the probability of the two inputs belonging to the same neighbourhood.
        """
        x_all = torch.cat([x, x_tild], -1)
        p = self.model(x_all)
        return p.view((-1,))

class TNC:
    '''The TNC model'''
    
    def __init__(
        self,
        args,
        config,
        device='cuda',
    ):
        '''
          Initialize a TNC model.

        '''
        
        self.args = args
        self.config = config
        super().__init__()
        
        self.device = device
        self.net = FeatureProjector(input_size=args['feature_dim'], output_size=args['out_features']).to(self.device)
        self.n_iters = 0

    def fit(self, train_dataset, ds_name, verbose=False):
        ''' Training the TNC model.
        
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
            run_name = 'TNC'

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
        loss_fn = torch.nn.BCEWithLogitsLoss()




        # Training and validation loop
        
        w = 0.1
        window_size = self.args['tnc_window']
        time_stamp = self.args['tnc_window']*self.args['batch_size']

        disc_model = Discriminator(self.args['out_features'], self.device)
        disc_model.to(self.device)
        
     

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
                
                x = x.to(self.device)
               
                # Flatten features to have dimensions [batch_size * sequence_length, feature dim]
                x = x.reshape(-1, x.size(-1))
                x = x.T
                
                t = np.random.randint(2*window_size, time_stamp-2*window_size)
                x_t = x[:,t-window_size//2:t+window_size//2]
                
                x_p, delta = self.find_neighours(x, t, window_size)
                x_n = self.find_non_neighours(x, t, delta, window_size)
                
                mc_sample = x_p.shape[0]
                x_t = torch.transpose(x_t, 1, 0).unsqueeze(0)
                
                x_p = torch.transpose(x_p, 2, 1)
                x_n = torch.transpose(x_n, 2, 1)
                
                x_t = x_t.repeat(mc_sample, 1, 1)
                
                neighbors = torch.ones((len(x_p)*window_size)).to(self.device)
                non_neighbors = torch.zeros((len(x_n)*window_size)).to(self.device)
                x_t, x_p, x_n = x_t.to(self.device), x_p.to(self.device), x_n.to(self.device)
                
                z_t = self.net(x_t)
                z_p = self.net(x_p)
                z_n = self.net(x_n)

                d_p = disc_model(z_t, z_p)
                d_n = disc_model(z_t, z_n)
                
                p_loss = loss_fn(d_p, neighbors)
                n_loss = loss_fn(d_n, non_neighbors)
                n_loss_u = loss_fn(d_n, neighbors)
                loss = (p_loss + w*n_loss_u + (1-w)*n_loss)/2
                
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
    
    def find_neighours(self, x, t, window_size):
    
        T = self.args['sequence_sample']
        mc_sample_size = self.args['tnc_window']
        adf = True
        
        if adf:
            gap = window_size
            corr = []
            for w_t in range(window_size,4*window_size, gap):
                try:
                    p_val = 0
                    for f in range(x.shape[-2]):
                        p = adfuller(np.array(x[f, max(0,t - w_t):min(x.shape[-1], t + w_t)].reshape(-1, )))[1]
                        p_val += 0.01 if math.isnan(p) else p
                    corr.append(p_val/x.shape[-2])
                except:
                    corr.append(0.6)
            epsilon = len(corr) if len(np.where(np.array(corr) >= 0.01)[0])==0 else (np.where(np.array(corr) >= 0.01)[0][0] + 1)
            delta = 5*epsilon*window_size

        ## Random from a Gaussian
        t_p = [int(t+np.random.randn()*epsilon*window_size) for _ in range(mc_sample_size)]
        t_p = [max(window_size//2+1,min(t_pp,T-window_size//2)) for t_pp in t_p]
        x_p = torch.stack([x[:, t_ind-window_size//2:t_ind+window_size//2] for t_ind in t_p])
        
        return x_p, delta

    def find_non_neighours(self, x, t, delta, window_size):
        T = self.args['sequence_sample']
        mc_sample_size = self.args['tnc_window']
        adf = True
        
        if t>T/2:
            t_n = np.random.randint(window_size//2, max((t - delta + 1), window_size//2+1), mc_sample_size)
        else:
            t_n = np.random.randint(min((t + delta), (T - window_size-1)), (T - window_size//2), mc_sample_size)
        x_n = torch.stack([x[:, t_ind-window_size//2:t_ind+window_size//2] for t_ind in t_n])

        if len(x_n)==0:
            rand_t = np.random.randint(0,window_size//5)
            if t > T / 2:
                x_n = x[:,rand_t:rand_t+window_size].unsqueeze(0)
            else:
                x_n = x[:, T - rand_t - window_size:T - rand_t].unsqueeze(0)
        return x_n

    def encode(self, x):
        self.net.eval()
        out = self.net(x.to(self.device))

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