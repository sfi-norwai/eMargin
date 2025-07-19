import torch
from models.contrastive import LS_HATCL_LOSS, HATCL_LOSS
from tqdm import tqdm
from src.models.attention_model import *
from src.models.mfclrencoder import *
from pytorch_lightning.loggers import WandbLogger
import wandb
from models.losses import hierarchical_contrastive_loss
from utils import take_per_row

from models.mfclrloss import combine_loss
from utils import take_per_row_multigrain, split_with_nan, centerize_vary_length_series, torch_pad_nan
from models.mfclrdata_aug import data_generation_multi


class MF_CLR:
    '''The MF_CLR model'''
    
    def __init__(
        self,
        args,
        config,
        device='cuda',
        max_train_length=None,
        temporal_unit=0,
        ph_dim = 320,
        hidden_dims=64,
        depth=10,
        projection= True,
        da= "proposed"
    ):
        '''
          Initialize a MF_CLR model.

        '''
        
        self.args = args
        self.config = config
        super().__init__()
        
        self.device = device
        self.n_iters = 0
        grain_split = [args['feature_dim'], args['feature_dim']]
        self.device = device

        if args['out_features'] is not None :
            self.output_dim_list = [args['out_features'] for i in range(len(grain_split) - 1)]
        else :
            self.output_dim_list = [320 for i in range(len(grain_split) - 1)]
        
        self.max_train_length = max_train_length
        self.temporal_unit = temporal_unit
        self.grain_split_list = grain_split      
        
        assert len(self.output_dim_list) == len(self.grain_split_list) - 1 
        self._net_list, self.net_list = [], []
        for l in range(len(grain_split) - 1):
            if len(self.net_list) == 0 :
                input_dim_l = args['feature_dim']
                output_dim_l = self.output_dim_list[0]
            else :
                if projection is False :
                    input_dim_l = self.output_dim_list[l - 1]
                    output_dim_l = self.output_dim_list[l]
                else :
                    input_dim_l = self.output_dim_list[l - 1] + ph_dim * l
                    output_dim_l = self.output_dim_list[l] + ph_dim * l
            self._net = BackboneEncoder(
                input_dims= input_dim_l, 
                output_dims= output_dim_l, 
                hidden_dims= hidden_dims, 
                depth= depth,
            ).to(self.device)            
            self.net = torch.optim.swa_utils.AveragedModel(self._net)
            self.net.update_parameters(self._net)
            self._net_list.append(self._net)
            self.net_list.append(self.net)
        
        self.ph_list = []
        for l in range(len(grain_split) - 1):
            projector = Projector(
                input_dims= grain_split[l + 1] - grain_split[l],
                h1= int(ph_dim / 2),
                h2= ph_dim,
            ).to(self.device)            
            self.ph_list.append(projector)

        self.projection = projection

        assert da in ["proposed", "scaling", "shifting", "jittering", "permutation", "random mask"]
        self.da_method = da
       
    
    def fit(self, train_dataset, ds_name, verbose=False):
        ''' Training the MF_CLR model.
        
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
            run_name = 'TS2Vec'

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
        
        
        optimizer_list, scheduler_list = [], []
        for g in range(len(self.grain_split_list) - 1):
            self.args['lr'] = float(self.args['lr'])
            optimizer = torch.optim.AdamW(list(self._net_list[g].parameters()) + list(self.ph_list[g].parameters()), lr= self.args['lr'])
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
            optimizer_list.append(optimizer)
            scheduler_list.append(scheduler)
      
        temporal_unit = 0
        max_train_length = 500

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
               
                if self.max_train_length is not None and x.size(1) > self.max_train_length :
                    window_offset = np.random.randint(x.size(1) - self.max_train_length + 1)
                    x = x[ : , window_offset : window_offset + self.max_train_length]
                x = x.to(self.device)

                ts_l = x.size(1)      
                grain_data_list = take_per_row_multigrain(x, np.random.randint(low= 0, high= 1, size= x.size(0)), ts_l, self.grain_split_list)


                for g in range(len(self.grain_split_list) - 1):
                    
                    if g == 0 :
                        fine_grained = grain_data_list[0]
                        coarse_grained = grain_data_list[1]
                    else :
                        fine_grained = out_embed
                        coarse_grained = grain_data_list[g + 1]


                    this_optimiser = optimizer
                    this_optimiser.zero_grad()

                    fine_grained = fine_grained.to(self.device)


    
                    out = self._net_list[g](fine_grained)
                    fine_grained_aug = torch.from_numpy(data_generation_multi(fine_grained.cpu().numpy())).float()
                    fine_grained_aug = fine_grained_aug.to(self.device)
                    out_aug = self._net_list[g](fine_grained_aug)
                    coarse_grained = coarse_grained.to(self.device)

                    grain_loss = combine_loss(out, out_aug, coarse_grained, coarse_grained)
                    grain_loss.backward()

                    train_running_loss += grain_loss.item()


                    fine_embed = out.clone().detach().cpu().numpy()
                    if self.projection is True :
                        coarse_embed = self.ph_list[g](coarse_grained)
                        coarse_embed = coarse_embed.clone().detach().cpu().numpy()
                        out_embed = np.concatenate([fine_embed, coarse_embed], axis= 2)
                    else :
                        out_embed = np.concatenate([fine_embed, coarse_embed], axis= 2)
                    out_embed = torch.from_numpy(out_embed)


                    this_optimiser.step()
                    self.net_list[g].update_parameters(self._net_list[g])
                    
                
                # Backward pass and optimization
                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()
                    
                # Update training statistics
                n_epoch_iters += 1
                self.n_iters += 1
                pbar.update(1)

                

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

        return out.transpose(1, 2)


    def save(self, fn):
        ''' Save the model to a file.
        
        Args:
            fn (str): filename.
        '''
        torch.save(self._net.state_dict(), fn)
    
    def load(self, fn):
        ''' Load the model from a file.
        
        Args:
            fn (str): filename.
        '''
        state_dict = torch.load(fn, map_location=self.device)
        self._net.load_state_dict(state_dict)