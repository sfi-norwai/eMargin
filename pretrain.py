import torch
import numpy as np
import argparse
import os
import time
import datetime
from baselines.emargin import eMargin
from baselines.cost import CoST
from baselines.ts2vec import TS2Vec
from baselines.timedrl import TimeDRL
from baselines.soft_ts2vec import Soft
from baselines.infots import InfoTS
from baselines.tnc import TNC
from baselines.simmtm import SimMTM
from baselines.mf_clr import MF_CLR
from utils import name_with_datetime, pkl_save
import wandb
import tasks
import argparse
import random
from baselines.cpc import CPC
import src.data
from src.loader.dataloader import STFTDataset, SLEEPDataset
from torch.utils.data import Subset





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Start Vanilla CL training.')
    parser.add_argument('model', help='The model name')
    parser.add_argument('dataset', help='The dataset name')
    parser.add_argument('-p', '--params_path', required=False, type=str,
                        help='params path with config.yml file',
                        default='configs/sleepconfig.yml')
    parser.add_argument('-s', '--seed_value', required=False, type=int,
                        help='seed value.', default=42)
    parser.add_argument('-b', '--batch_size', required=False, type=int,
                        help='seed value.', default=8) 
    parser.add_argument('-v', '--verbose_bool', required=False, type=bool,
                        help='verbose bool.', default=False)
    parser.add_argument('-g', '--gpu', required=False, type=int,
                        help='int.', default=0)
    parser.add_argument('-th', '--max_threads', required=False, type=int,
                        help='number of threads.', default=8)
    parser.add_argument('-iter', '--iterations', required=False, type=int,
                        help='number of iterations.', default=None)
    parser.add_argument('--evaluate', required=False, type=str,
                        help='Task to evaluate on.', default=None)
    
    parser.add_argument('-sp', '--semi_percentage', required=False, type=int,
                        help='percentage of training data.', default=0.01)
    
    parser.add_argument('-trans', '--transfer_data', required=False, type=str,
                        help='The transfer dataset name', default='ecg')
    
   
    
    
    
    pargs = parser.parse_args()
    config_path = pargs.params_path
    # Read config
    config = src.config.Config(config_path)
    config.SEED = pargs.seed_value
    ds_path = pargs.dataset
    verbose = pargs.verbose_bool
    gpu_val = pargs.gpu
    max_threads = pargs.max_threads
    
    # Log in to Wandb
    if config.WANDB:
        wandb.login(key=config.WANDB_KEY)
    
    for ds_args in src.utils.grid_search(config.DATASET_ARGS):
        # Iterate over all model configs if given
        for args in src.utils.grid_search(config.ALGORITHM_ARGS):
            
            seed = config.SEED
            if pargs.iterations is not None:
                 args['iterations'] = pargs.iterations

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

            # Set all seeds:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # Multi-GPU
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
            
            # Create the dataset
            if config.DATASET == 'STFT':
                dataset = src.data.get_dataset(
                        dataset_name=config.DATASET,
                        dataset_args=ds_args,
                        root_dir=f'datasets/{ds_path}',
                        num_classes=config.num_classes,
                        label_map=config.label_index,
                        replace_classes=config.replace_classes,
                        config_path=config.CONFIG_PATH,
                        name_label_map=config.class_name_label_map
                    )
                
            elif config.DATASET == 'ECG':
                dataset = STFTDataset(
                        data_path=f'datasets/{ds_path}',
                        n_fft = ds_args['n_fft'],
                        seq_length=ds_args['seq_length'],
                        class_to_exclude=ds_args['class_to_exclude'],
                        hop_length=ds_args['hop_length'],
                        win_length=ds_args['win_length'],
                        num_labels=ds_args['num_labels']
                    )
            
            elif config.DATASET == 'SLEEPEEG':
                train_ds = SLEEPDataset(
                        data_path=f'datasets/{ds_path}',
                        n_fft = ds_args['n_fft'],
                        seq_length=ds_args['seq_length'],
                        hop_length=ds_args['hop_length'],
                        win_length=ds_args['win_length'],
                        num_labels=ds_args['num_labels'],
                        eval=False
                    )
                valid_ds = SLEEPDataset(
                        data_path=f'datasets/{ds_path}',
                        n_fft = ds_args['n_fft'],
                        seq_length=ds_args['seq_length'],
                        hop_length=ds_args['hop_length'],
                        win_length=ds_args['win_length'],
                        num_labels=ds_args['num_labels'],
                        eval=True
                    )

            else:
                raise ValueError(f"Unsupported DATASET: {config.DATASET}")
            
            if not config.DATASET == 'SLEEPEEG':
                
                valid_split = config.VALID_SPLIT
                
                valid_amount = int(np.floor(len(dataset)*valid_split))
                train_amount = len(dataset) - valid_amount
                
                train_indices = list(range(train_amount))
                valid_indices = list(range(train_amount, train_amount + valid_amount))
                
                # Create subsets
                train_ds = Subset(dataset, train_indices)
                valid_ds = Subset(dataset, valid_indices)

            run_dir = 'training/' + pargs.dataset + '_' + str(config.SEED) + '_' + name_with_datetime(pargs.model)
            os.makedirs(run_dir, exist_ok=True)
            
            t = time.time()
            
            if pargs.model == 'eMargin':
                model = eMargin(
                    args,
                    config,
                    device=device
                )

            elif pargs.model == 'TS2Vec':
                model = TS2Vec(
                    args,
                    config,
                    device=device
                )

            elif pargs.model == 'TimeDRL':
                model = TimeDRL(
                    args,
                    config,
                    device=device
                )

            elif pargs.model == 'CoST':
                model = CoST(
                    args,
                    config,
                    device=device
                )
            
            elif pargs.model == 'Soft':
                model = Soft(
                    args,
                    config,
                    device=device
                )

            elif pargs.model == 'InfoTS':
                model = InfoTS(
                    args,
                    config,
                    device=device
                )

            elif pargs.model == 'TNC':
                model = TNC(
                    args,
                    config,
                    device=device
                )

            elif pargs.model == 'SimMTM':
                model = SimMTM(
                    args,
                    config,
                    device=device
                )

            elif pargs.model == 'CPC':
                model = CPC(
                    args,
                    config,
                    device=device
                )

            elif pargs.model == 'MF_CLR':
                model = MF_CLR(
                    args,
                    config,
                    device=device
                )

            else:
                raise ValueError(f"Unsupported BASELINE: {pargs.model}")

            loss_log = model.fit(
                train_ds,
                ds_path,
                verbose=pargs.verbose_bool
            )
            model.save(f'{run_dir}/model.pkl')

            t = time.time() - t
            print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")

            if pargs.evaluate:
                if pargs.evaluate == 'supervised':
                    eval_res = tasks.supervised_evaluation(model, train_ds, valid_ds, args['out_features'], args['linear_epochs'], args['batch_size'], config)
                elif pargs.evaluate == 'semi_supervised':
                    eval_res = tasks.semi_supervised_evaluation(model, train_ds, valid_ds, args['out_features'], args['linear_epochs'], args['batch_size'], pargs.semi_percentage/100, config)
                elif pargs.evaluate == 'clustering':
                    eval_res = tasks.clustering_evaluation(model, valid_ds, config)
                else:
                    assert False
                pkl_save(f'{run_dir}/eval_res.pkl', eval_res)
                print('Evaluation result:', eval_res)

    print("Finished.")
