from torch.nn import GRU, Linear, CrossEntropyLoss
import wandb

import torch
from models.contrastive import LS_HATCL_LOSS, HATCL_LOSS
from tqdm import tqdm
from src.models.attention_model import *
from pytorch_lightning.loggers import WandbLogger
import wandb
import numpy as np

class CPC:
    '''The CPC model'''
    
    def __init__(
        self,
        args,
        config,
        device='cuda',
    ):
        '''
          Initialize a CPC model.

        '''
        
        self.args = args
        self.config = config
        super().__init__()
        
        self.device = device
        self.net = FeatureProjector(input_size=args['feature_dim'], output_size=args['out_features']).to(self.device)
        self.n_iters = 0
        

       
    
    def fit(self, train_dataset, ds_name, verbose=False):
        ''' Training the CPC model.
        
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
            run_name = 'CPC'

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
        
        acc = 0
        ds_estimator = Linear(self.args['out_features'], self.args['out_features']).to(self.device)
        auto_regressor = GRU(input_size=self.args['out_features'], hidden_size=self.args['out_features'], batch_first=True).to(self.device)

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
    
                x = x.to(self.device) 

                # Create a mask tensor with the same shape as the original tensor
                n_size = x.shape[1]//2
                sequence_length = x.shape[1]
                
                # Encode the entire sequence
                encodings = self.net(x)  # [batch_size, seq_len, encoding_size]

                # Choose a fixed position, e.g., the center of the sequence
                center_index = sequence_length // 2
                selected_encodings = encodings[:, center_index:center_index+5, :]  # [batch_size, 1, encoding_size]

                # Pass through GRU
                _, c_t = auto_regressor(selected_encodings)  # c_t: [1, batch_size, encoding_size]

                # Project the context vector c_t using ds_estimator
                c_t_projected = ds_estimator(c_t.squeeze(0))  # [batch_size, encoding_size]

                # Calculate density ratios
                density_ratios = torch.bmm(encodings, c_t_projected.unsqueeze(-1)).squeeze(-1)  # [batch_size, seq_len]

                # Select random negative samples and the positive sample
                rnd_n = np.random.choice(list(range(0, center_index - 2)) + list(range(center_index + 3, sequence_length)), n_size)
    
                X_N = torch.cat([density_ratios[:, rnd_n], density_ratios[:, center_index + 1].unsqueeze(1)], dim=1) 

                # Check if the correct index is chosen
                acc += (torch.argmax(X_N, dim=1) == len(X_N[0]) - 1).sum().item()

                # Create labels that match the batch size
                labels = torch.full((X_N.size(0),), len(X_N[0]) - 1, dtype=torch.long).to(self.device)

                # Calculate the loss
                loss = CrossEntropyLoss()(X_N, labels)
                
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