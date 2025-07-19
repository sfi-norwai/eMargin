
# from src.losses.contrastive import SimMTMLoss
import torch
from models.contrastive import LS_HATCL_LOSS, HATCL_LOSS
from tqdm import tqdm
from src.models.attention_model import *
from pytorch_lightning.loggers import WandbLogger
import wandb
from models import TSEncoder

from src.losses.simmtm_losses import *
from src.augmentations import data_transform_masked4cl

class SimMTM:
    '''The SimMTM model'''
    
    def __init__(
        self,
        args,
        config,
        device='cuda',
    ):
        '''
          Initialize a SimMTM model.

        '''
        
        self.args = args
        self.config = config
        super().__init__()
        
        self.device = device
        self.net = FeatureProjector(input_size=args['feature_dim'], output_size=args['out_features']).to(self.device)
        self.head = nn.Linear(args['out_features'], args['feature_dim']).to(self.device)

        self.awl = AutomaticWeightedLoss(2)
        self.contrastive = ContrastiveWeight(temperature=0.2)
        self.aggregation = AggregationRebuild(temperature=0.2)
        self.mse = torch.nn.MSELoss()

        # self.net = TSEncoder(input_dims=args['feature_dim'], output_dims=args['out_features']).to(self.device)
        self.n_iters = 0

        
        

       
    
    def fit(self, train_dataset, ds_name, verbose=False):
        ''' Training the SimMTM model.
        
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
            run_name = 'SimMTM'

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

        n_iters = self.args['iterations']
        pbar = tqdm(total=n_iters, desc="Training")
        epoch = 0

        while True:
            # Training phase
            self.net.train()  # Set the model to training mode
            train_running_loss = 0.0
            n_epoch_iters = 0

            for data, _ in train_loader:

                interrupted = False
                if n_iters is not None and self.n_iters >= n_iters:
                    interrupted = True
                    break
                
                data_masked_m, mask = data_transform_masked4cl(data, 0.5, 3, 3)
                data_masked_om = torch.cat([data, data_masked_m], 0)
                data = data.float().to(self.device)
                data_masked_om = data_masked_om.float().to(self.device)
        
                optimizer.zero_grad()
                
                x = self.net(data_masked_om)
                x_in_t = data_masked_om
                
                z = x.reshape(-1, x.size(-1))

                # loss = self.loss(output, target)
                loss_cl, similarity_matrix, logits, positives_mask = self.contrastive(z)
                rebuild_weight_matrix, agg_x = self.aggregation(similarity_matrix, z)
                pred_x = self.head(agg_x.reshape(agg_x.size(0), -1))

                loss_rb = self.mse(pred_x, x_in_t.reshape(-1, x_in_t.size(-1)).detach())
                loss = self.awl(loss_cl, loss_rb)
                
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






