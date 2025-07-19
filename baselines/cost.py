import torch
from tqdm import tqdm
from src.models.costmodel import *
from src.models.costencoder import *
from src.utils import take_per_row
import src.config, src.utils, src.models, src.hunt_data
from pytorch_lightning.loggers import WandbLogger
import wandb
import numpy as np
from tqdm import tqdm
import math
from pytorch_lightning.loggers import WandbLogger
import wandb

class CoST:
    '''The CoST model'''
    
    def __init__(
        self,
        args,
        config,
        device='cuda',
    ):
        '''
          Initialize a CoST model.

        '''
        
        self.args = args
        self.config = config
        super().__init__()
        
        self.device = device

        net = CoSTEncoder(
                input_dims=args['feature_dim'], output_dims=args['out_features'],
                kernels=[1, 2, 4, 8, 16, 32, 64, 128],
                length=args['sequence_sample'],
                hidden_dims=args['feature_dim'], depth=10,
            )
        
        self.net = CoSTModel(
                net,
                net,
                kernels=[1, 2, 4, 8, 16, 32, 64, 128],
                dim=args['out_features'],
                alpha=0.05,
                K=256,
                device = device,
            ).to(device)

        self.n_iters = 0
        

       
    
    def fit(self, train_dataset, ds_name, verbose=False):
        ''' Training the CoST model.
        
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
            run_name = 'CoST'

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
        
        # Define optimizer

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
        
        max_train_length = 300
        my_transform = CoST_Transform(sigma=0.5, multiplier=5)

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
               
                x_q, x_k =  x.to(self.device), my_transform.transform(x).to(self.device)

                if max_train_length is not None and x_q.size(1) > max_train_length:
                    window_offset = np.random.randint(x_q.size(1) - max_train_length + 1)
                    x_q = x_q[:, window_offset : window_offset + max_train_length]
                    x_k = x_k[:, window_offset : window_offset + max_train_length]

                loss = self.net(x_q, x_k)
                
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
        out = self.net.encoder_q.feature_extractor(x.to(self.device))

        return out


    def save(self, fn):
        ''' Save the model to a file.
        
        Args:
            fn (str): filename.
        '''
        torch.save(self.net.encoder_q.feature_extractor.state_dict(), fn)
    
    def load(self, fn):
        ''' Load the model from a file.
        
        Args:
            fn (str): filename.
        '''
        state_dict = torch.load(fn, map_location=self.device)
        self.net.encoder_q.feature_extractor.load_state_dict(state_dict)
