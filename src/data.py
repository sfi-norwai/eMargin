import os
import pathlib
import torch
import einops
import random
import pandas as pd
import numpy as np
import functools
import datetime
from tqdm import tqdm
import src.utils
from src.utils import MomentEstimator
from collections.abc import Iterable
from scipy.spatial.transform import Rotation as R
import zipfile
import tempfile

import sys


def get_dataset(
    dataset_name,
    dataset_args,
    root_dir,
    config_path,
    num_classes=None,
    skip_files=[],
    label_map=None,
    replace_classes=None,
    test_mode=False,
    valid_mode=False,
    inference_mode=False,
    name_label_map=None
):
    allowed_datasets = ['TimeSeries', 'STFT', 'HUNT4Masked',
                        'USCHAD', 'PAMAP2', 'MobiAct']
    
    if dataset_name in allowed_datasets:
        cls = getattr(
            sys.modules[__name__],
            f'{dataset_name}Dataset'
        )
        return cls(args=dataset_args,
                   root_dir=root_dir,
                   config_path=config_path,
                   num_classes=num_classes,
                   skip_files=skip_files,
                   label_map=label_map,
                   replace_classes=replace_classes,
                   test_mode=test_mode,
                   valid_mode=valid_mode,
                   inference_mode=inference_mode,
                   name_label_map=name_label_map
                  )
    else:
        raise ValueError((f'No Dataset class with name"{dataset_name}".\n'
                          f'Allowed dataset names: {allowed_datasets}'))


class HARDataset(torch.utils.data.Dataset):
    def __init__(
        self, root_dir,
        x_columns, y_column,
        num_classes,
        padding_val=0.0,
        label_map=None,
        replace_classes=None,
        config_path='',
        skip_files=[],
        test_mode=False,
        valid_mode=False,
        inference_mode=False,
        sep=',',
        header='infer',
    ):
        '''Super class dataset for classification

        Parameters
        ----------
        root_dir (string): Directory of training data
        x_columns (list of str): columns of sensors
        y_column (str): column of label
        num_classes (int): how many classes in dataset
        padding_val (float), optional: Value to insert if padding required
            Note: padding is only applied in inference mode to avoid label
            padding.
        label_map (dict): mapping labels in dataframe to other values
        replace_classes (dict): mapping which replace to replace
        config_path (str): to save normalization params on disk
        skip_files (list of string):
            csv files to in root_dir not to use. If None, all are used
        test_mode (bool): whether dataset used for testing
        valid_mode (bool): whether dataset used for validation
        inference_mode (bool): whether dataset used for inference
            i.e., y is not returned
        sep (str): Which sep used in dataset files
        header (int, list of int, None): Row numbers to use as column names

        '''
        self._cache = {}
        self.root_path = root_dir
        self.x_columns = x_columns
        self.y_column = y_column
        self.num_classes = num_classes
        self.padding_val = padding_val
        self.label_map = label_map
        self.replace_dict = replace_classes
        self.config_path = config_path
        self.skip_files = skip_files
        self.train_mode = not (test_mode or inference_mode or valid_mode)
        self.inference_mode = inference_mode
        self.sep = sep
        self.header = header

    def replace_classes(self, y):
        if self.replace_dict:
            return y.replace(self.replace_dict)
        else:
            return y 

    #@property
    def y(self):
        '''Returns y/true_label tensor(s)

        Returns
        -------
        either tensor or dict of tensors with filename as keys

        '''
        msg = ('Implement y(): Returns y/true_label tensor(s)')
        raise NotImplementedError(msg)

    def post_proc_y(self, t, overlap_kind='mean'):
        '''Undo all changes made in this Dataset class to original y data

        It assumes tensor with shape
        [num_batches, sequence_length, d]
        Depending on subclass, different operations have to
        be performed to achieve correct alignement
        d can be any dimension
        Example: model probability prediction with shape
                [num_batches,sequence_length,num_classes]

        Parameters
        ----------
        t (tensor):
            Has shape [num_batches,sequence_length,d]
        overlap_kind (str), optional:
            What to do with possible overlapping areas. (default is 'mean')
            'sum' adds the values in the overlapping areas
            'mean' computes the mean of the overlapping areas

        Returns
        -------
        either tensor or dict of tensors with filename as keys
            Each tensor's shape: [signal_len, n]

        '''
        msg = ('Implement post_proc_y(): Returns tensor(s) '
               'aligned with original signal')
        raise NotImplementedError(msg)


 
    def _label_cols_available(self):
        '''Is there a y_column in every given root file'''
        filenames = [x for x in os.listdir(self.root_path) \
                     if x not in self.skip_files]
        for fn in tqdm(filenames):
            available_cols = pd.read_csv(
                os.path.join(self.root_path, fn),
                index_col=0,
                nrows=0,
                header=self.header,
                sep=self.sep,
            ).columns.tolist()
            if self.y_column not in available_cols:
                print(f'No label column {self.y_column} in {fn}...'
                      'Skipping labels')
                return False
        return True





class STFTDataset(HARDataset):
    """Dataset for spectrogram-based HAR."""

    def __init__(self, args,
                 root_dir,
                 num_classes,
                 config_path='',
                 label_map=None,
                 replace_classes=None,
                 skip_files=[],
                 test_mode=False,
                 valid_mode=False,
                 inference_mode=False,
                 **kwargs
        ):
        '''Using spectrograms of time series signals as dataset

        Parameters
        ----------
        root_dir (string): Directory of training data
        x_columns (list of str): columns of sensors
        y_column (str): column of label
        num_classes (int): how many classes in dataset
        padding_val (float), optional: Value to insert if padding required
            Note: padding is only applied in inference mode to avoid label
            padding.
        label_map (dict): mapping labels in dataframe to other values
        replace_classes (dict): mapping which classes to replace
        config_path (str): to save normalization params on disk
        skip_files (list of string):
            csv files to in root_dir not to use. If None, all are used
        test_mode (bool): whether dataset used for testing
        valid_mode (bool): whether dataset used for validation
        inference_mode (bool): whether dataset used for inference
            i.e., y is not returned
        args (dict): Dataset specific parameters
            Needs to include:
                n_fft (int): STFT window size
                hop_length (int): STFT window shift
                normalize (bool): Whether to normalize spectrograms
                phase (bool): Include phase

        '''
        self.normalize = args['normalize']
        self.stack_axes = args['stack_axes'] if 'stack_axes' in args else True
        self.unstack_sensors = args['unstack_sensors'] if 'unstack_sensors' in args else False
        self.size = 0
        self._y = {}
        # Read file params
        self.drop_labels = args['drop_labels'] if 'drop_labels' in args else []
        # In case of resampling
        self.source_freq = args['source_freq'] if 'source_freq' in args else 50
        self.target_freq = args['target_freq'] if 'target_freq' in args else 50
        # Freq domain split
        self.n_fft = args['n_fft']
        self.hop_length = args['hop_length']
        self.hop_length = self.n_fft//2 if args['hop_length'] is None else args['hop_length']
        self.window = torch.hann_window(self.n_fft)
        self.phase = args['phase']
        # Time domain split
        assert args['sequence_length'] > self.hop_length, \
                print('sequence_length < hop_length not allowed')
        self.sequence_length = args['sequence_length'] // self.hop_length -1
        frame_shift = args['frame_shift']
        if frame_shift is None:
            frame_shift = args['sequence_length']
        elif frame_shift == 'half':
            frame_shift = args['sequence_length']//2
        self.frame_shift = frame_shift // self.hop_length
        
        # Windowed labels handling
        self.windowed_labels_kind = args['windowed_labels_kind'] \
                if 'windowed_labels_kind' in args else 'argmax'
        super().__init__(
            root_dir=root_dir,
            config_path=config_path,
            x_columns=args['x_columns'],
            y_column=args['y_column'],
            padding_val=args['padding_val'],
            num_classes=num_classes,
            label_map=label_map,
            replace_classes=replace_classes,
            skip_files=skip_files,
            test_mode=test_mode,
            valid_mode=valid_mode,
            inference_mode=inference_mode,
            sep=args['sep'] if 'sep' in args else ',',
            header=args['header'] if 'header' in args else 'infer',
        )
      
        self.data = self.read_all(root_dir)
        self.data_ranges = self._get_data_ranges()
        if self.normalize:
            if 'norm_params_path' in args:
                self.normalize_params_path = args['norm_params_path']
            else:
                self.normalize_params_path = os.path.join(
                    config_path,
                    f'normalization_params_STFT_feats{self.feature_dim}_seqlen{self.seq_length}'
                )
            force = args['force_norm_comp'] if 'force_norm_comp' in args else False
            force = force and self.train_mode  # Force impossible for test/valid
            self.mean = self._mean(save_on_disk=self.train_mode, force=force)
            self.std = self._std(save_on_disk=self.train_mode, force=force)
            self.normalize_data()

    def __getitem__(self, idx):
        fn = self.get_filename_for_idx(idx)
        # Identify idx in dataframe
        range_start_idx = min(self.data_ranges[fn])
        start_idx = idx-range_start_idx
        start_idx = start_idx * self.frame_shift
        end_idx = start_idx + self.sequence_length
        win_len = end_idx - start_idx
        # Determine window to return:
        x = self.data[fn][0][start_idx:end_idx]
        
        if self.inference_mode:
            if len(x) != len(self.data[fn][0]):
                # In inference_mode padding is applied, otherwise shape mismatch
                overflow = abs(min(0, len(self.data[fn][0])-end_idx))
                x = torch.nn.functional.pad(
                    input=x,
                    pad=[0,0,0,overflow],
                    value=self.padding_val
                )
                if self.unstack_sensors:
                    x = self._unstack_sensors(x)
            return x
        else:
            y = self.data[fn][1][start_idx:end_idx]
            if self.unstack_sensors:
                x = self._unstack_sensors(x)
       
            return x, y

    def __len__(self):
        return self.size

    @property
    def seq_length(self):
        '''Input sequence length'''
        return self.sequence_length

    @property
    def feature_dim(self):
        '''Input feature dimensionality'''
        if self.stack_axes:
            base_shape = (self.n_fft // 2 + 1) * len(self.x_columns)
        else:
            base_shape = self.n_fft // 2 + 1
        if self.unstack_sensors:
            num_sensors = len(self.x_columns)//3
            base_shape = base_shape // num_sensors
        if not self.phase:
            return base_shape
        else:
            return base_shape * 2

    @property
    def output_shapes(self):
        '''Shape of y output if given and one-hot encoded'''
        return self.num_classes

    @property
    def input_shape(self):
        '''Num bins'''
        return self.feature_dim

    #@property
    def y(self, return_probs=False, probs_aggr_window_len=None):
        '''Returns y_column values as indices or probabilities

        Parameters
        ----------
        return_probs (bool, optional): Compute probabilities
        probs_aggr_window_len (int, optional): Window length for probs

        Returns
        -------
        dict of tensors

        '''
        if return_probs:
            if probs_aggr_window_len:
                aggr_len = probs_aggr_window_len
                aggr_shift = probs_aggr_window_len
            else:
                aggr_len = self.n_fft
                aggr_shift = self.hop_length
            new_y = {}
            for fn, y_true in self._y.items():
                new_y[fn] = windowed_labels(
                    labels=y_true,
                    num_labels=self.num_classes,
                    frame_length=aggr_len,
                    frame_step=aggr_shift,
                    pad_end=True,
                    kind='density'
                )
            return new_y
        return self._y

    def post_proc_y(self, t, overlap_kind='mean', return_probs=False, probs_aggr_window_len=None):
        '''Undo all changes made in this Dataset to original y data

        Here, sliding windows are aligned 2 or 3 times:
        1 for normal splitting, 1 for STFT computation, and 1 if resampling done.
        argmax applied to probabilities to get class indices values.

        Parameters
        ----------
        t (array like): tensor to process
        overlap_kind (str, optional): How to handle overlaps when unfolding
        return_probs (bool, optional): Do not apply argmax if True
        probs_aggr_window_len (int, optional): aggregate probs if not None

        '''
        t_dict = {}
        for filename, _range in self.data_ranges.items():
            try:
                _t = t[_range].numpy()
            except TypeError:
                _t = np.array(t[slice(_range.start,_range.stop)][0])
            # Split to spectrograms dim
            if _t.shape[0] != 1:
                _t = unfold_windows(
                    arr=_t,
                    window_size=self.sequence_length,
                    window_shift=self.frame_shift,
                    overlap_kind=overlap_kind
                )
            else:
                _t = _t[0]
            if self.inference_mode:
                # Cut padded parts at the right side
                overflow = abs(min(0, len(self.data[filename][0])-len(_t)))
                _t = _t[:len(_t)-overflow]
            if return_probs:
                if probs_aggr_window_len:
                    amount_to_inlude = get_num_slices(
                        total_amount=probs_aggr_window_len,
                        sequence_length=self.n_fft,
                        frame_shift=self.hop_length,
                        padding=False
                    )
                    amount_to_shift = get_num_slices(
                        total_amount=probs_aggr_window_len,
                        sequence_length=self.n_fft,
                        frame_shift=self.hop_length,
                        padding=True
                    )
                    new_t = []
                    for i in range(0, len(_t), amount_to_shift):
                        cutted_probs = _t[i:i+amount_to_inlude]
                        new_t.append(cutted_probs.mean(axis=0))
                    _t = np.array(new_t)
                t_dict[filename] = _t
                continue
            # Spectrograms dim to time dim
            _t = unfold_windows(
                arr=_t,
                window_size=self.n_fft,
                window_shift=self.hop_length,
                overlap_kind=overlap_kind
            )
            # Undo resampling if required
            if self.source_freq != self.target_freq:
                df_t = pd.DataFrame(_t)
                _t = src.utils.resample(
                    signal=df_t,
                    source_rate=self.target_freq,
                    target_rate=self.source_freq,
                    discrete_columns=df_t.columns,
                    resampler='fourier',
                    padder=None,
                    pad_size=None
                ).values
            if self.inference_mode:
                # Cut padded parts at the end
                t_dict[filename] = _t[:self.y()[filename].shape[0]]
            else:
                t_dict[filename] = _t
        if not return_probs:
            t_dict = src.utils.argmax(t_dict, axis=-1)  # Get classes for preds
        return t_dict

    def read_all(self, root_path):
        """ Reads all csv files in a given path and computes STFT"""
        data = {}
        filenames = [x for x in os.listdir(root_path) \
                     if x not in self.skip_files]
        uc = self.x_columns+[self.y_column]
        for fn in tqdm(filenames):
            df = pd.read_csv(
                os.path.join(root_path, fn),
                sep=self.sep,
                usecols=uc,
                header=self.header,
            )
            for drop_label in self.drop_labels:
                df = df[df[self.y_column]!=drop_label]
            df = df.dropna()  # Drop nan values
            # Required for classification
            if self._label_cols_available:
                df[self.y_column] = self.replace_classes(df[self.y_column])
            if self._label_cols_available and self.label_map is not None:
                df[self.y_column] = df[self.y_column].apply(
                    lambda _x: self.label_map[_x]
                )
                self._y[fn] = df[self.y_column].values
            # Resampling if required
            if self.source_freq != self.target_freq:
                discrete_columns=[self.y_column]
                df = src.utils.resample(
                    signal=df,
                    source_rate=self.source_freq,
                    target_rate=self.target_freq,
                    discrete_columns=discrete_columns,
                    resampler='fourier',
                    padder=None,
                    pad_size=None
                )
            x = torch.tensor(df[self.x_columns].values,dtype=torch.float32)
            # reshape required for correct STFT computation:
            # [signal_len, num_channels] -> [num_channels, signal_len]
            x = einops.rearrange(x, 'S C -> C S')
            if self.inference_mode:
                # Padding to make STFT computation easier
                overflow = np.floor((x.shape[-1]-1)/self.hop_length)
                overflow = int(overflow*self.hop_length + self.n_fft)
                overflow = abs(min(0, x.shape[-1]-overflow))
                x = torch.nn.functional.pad(
                    input=x,
                    pad=[0,overflow],
                    value=self.padding_val
                )
            
            x = torch.stft(
                input=x,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=self.window,
                center=False,
                return_complex=True
            )  # [num_channels, num_bins, num_frames]
            x_cartesian = src.utils.complex_to_cartesian(x)
            x_magnitude = src.utils.complex_to_magnitude(x, expand=True)
            
            
            x = x_cartesian if self.phase else x_magnitude
            if self.stack_axes:
                # Stack all spectrograms and put time dim first:
                # [num_channels, num_bins, num_frames, stft_parts] ->
                # [num_frames, num_channels x num_bins x stft_parts]
                x = einops.rearrange(x, 'C F T P -> T (C F P)')  # P=2
            else:
                x = einops.rearrange(x, 'C F T P -> T C F P')
            
            if self._label_cols_available:
                y = windowed_labels(
                    labels=df[self.y_column].values,
                    num_labels=self.num_classes,
                    frame_length=self.n_fft,
                    frame_step=self.hop_length,
                    pad_end=self.inference_mode,
                    kind=self.windowed_labels_kind
                )
                y_dtype = torch.int64 \
                        if self.windowed_labels_kind=='argmax' \
                        else torch.float32
                y = torch.tensor(y, dtype=y_dtype)
                data[fn] = (x, y)
            else:
                data[fn] = (x, None)
        return data

    def _get_data_ranges(self):
        '''To identify which subj to use given idx'''
        data_ranges = {}
        for fn, (x, y) in self.data.items():
            num_slices = get_num_slices(
                total_amount=len(x),
                sequence_length=self.sequence_length,
                frame_shift=self.frame_shift,
                padding=self.inference_mode
            )
            self.size += num_slices
            data_ranges[fn] = range(self.size-num_slices, self.size)
        return data_ranges

    def _unstack_sensors(self, t):
        rt = []
        num_sensors = len(self.x_columns)//3
        for i in range(num_sensors):
            rt.append(t[:,i*self.feature_dim:(i+1)*self.feature_dim])
        return torch.stack(rt)

    def get_filename_for_idx(self, idx):
        '''Given idx, which filename to use'''
        return [fn for fn, r in self.data_ranges.items() if idx in r][0]


    def normalize_data(self):
        '''Normalize time signals'''
        for fn, (x,y) in self.data.items():
            x = normalize(x=x, mean=self.mean, std=self.std)
            self.data[fn] = (x,y)


    def _mean(self, save_on_disk=False, force=False):
        '''Mean across all samples for each feature

        If mean not already saved on disk in self.normalize_params_path,
        it is computed using self.data. Otherwise, it is read from
        disk.

        Parameters
        ----------
        save_on_disk (bool): Stores computed mean in normalize_params_path
        force (bool): Force recomputation of mean even if saved on disk

        Returns
        -------
        torch.Tensor

        '''
        _m_path = os.path.join(self.normalize_params_path, 'mean.csv')
        if not os.path.exists(_m_path) or force:
            if not self.train_mode:
                raise FileNotFoundError(
                    f'No normalization param found {_m_path}'
                )
            else:
                print('Creating mean...')
                _sum = sum([x.sum(axis=0) for x,_ in self.data.values()])
                _len = sum([x.shape[0] for x,_ in self.data.values()])
                _m = _sum/_len
                if save_on_disk:
                    # On disk we save the stacked version
                    _m_to_store = _m if self.stack_axes \
                            else einops.rearrange(_m,  'C F P -> (C F P)')
                    src.utils.store_tensor(_m_to_store, _m_path)
        else:
            _m = src.utils.load_tensor(_m_path)
            new_shape = (len(self.x_columns), self.feature_dim, 1)
            _m = _m if self.stack_axes \
                    else torch.reshape(_m, new_shape)
        return _m

    def _std(self, save_on_disk=False, force=False):
        '''Std across all samples for each feature

        If std not already saved on disk in self.normalize_params_path,
        it is computed using self.data. Otherwise, it is read from
        disk.

        Parameters
        ----------
        save_on_disk (bool): Stores computed std in normalize_params_path
        force (bool): Force recomputation of std even if saved on disk

        Returns
        -------
        torch.Tensor

        '''
        _s_path = os.path.join(self.normalize_params_path, 'std.csv')
        if not os.path.exists(_s_path) or force:
            if not self.train_mode:
                raise FileNotFoundError(
                    f'No normalization param found {_s_path}'
                )
            else:
                print('Creating std...')
                _m = self.mean
                _sum = sum([((x-_m)**2).sum(axis=0) for x,_ in self.data.values()])
                _len = sum([x.shape[0] for x,_ in self.data.values()])
                _s = np.sqrt(_sum/_len)
                if save_on_disk:
                    # On disk we save the stacked version
                    _s_to_store = _s if self.stack_axes \
                            else einops.rearrange(_s,  'C F P -> (C F P)')
                    src.utils.store_tensor(_s_to_store, _s_path)
        else:
            _s = src.utils.load_tensor(_s_path)
            new_shape = (len(self.x_columns), self.feature_dim, 1)
            _s = _s if self.stack_axes \
                    else torch.reshape(_s, new_shape)
        _s = torch.where(_s==0.0, EPS, _s)  # Avoid div by 0
        return _s

    def collate_fn(self, data):
        '''Custom collate_fn for different sequence lengths in a batch'''
        x = torch.nn.utils.rnn.pad_sequence([d[0] for d in data],
                                            batch_first=True,
                                            padding_value=0.0)
        y = torch.nn.utils.rnn.pad_sequence([d[1] for d in data],
                                            batch_first=True,
                                            padding_value=0)
        # 0s where padding applied
        mask = torch.ones(y.shape[:2])
        for i in range(len(mask)):
            mask[i][len(data[i][1]):] = 0.0
        return [x, y, mask]



def get_num_slices(
    total_amount,
    sequence_length,
    frame_shift=1,
    padding=False
):
    '''Number of windows with frame shift in sliding window

    Parameters
    ----------
    total_amount (int)
    sequence_length (int)
    frame_shift (int), optional
    padding (bool), optional:
        If total_amount cannot be split perfectly into equaly sized
        windows, shall the last window be removed (keep_last=False)
        or not (keep_last=True)? (Default: False)

    Returns
    -------
    (int): Number of slices a tensor of length total_amount can be
        divided given sequence_length and frame_shift

    '''
    if padding:
        return max(1, int(np.ceil(total_amount/frame_shift)))
    else:
        return max(1, int(np.ceil((total_amount+1-sequence_length)/frame_shift)))


def windowed_labels(
    labels,
    num_labels,
    frame_length,
    frame_step=None,
    pad_end=False,
    kind='density',
):
    """Generates labels that correspond to STFTs

    With kind=None we are able to split the given labels
    array into batches. (T, C) -> (B, T', C)

    Parameters
    ----------
    labels : np.array

    Returns
    -------
    np.array
    """
    # Labels should be a single vector (int-likes) or kind has to be None
    labels = np.asarray(labels)
    if kind is not None and not labels.ndim == 1:
        raise ValueError('Labels must be a vector')
    if not (labels >= 0).all():
        raise ValueError('All labels must be >= 0')
    if not (labels < num_labels).all():
        raise ValueError(f'All labels must be < {num_labels} (num_labels)')
    # Kind determines how labels in each window should be processed
    if not kind in {'counts', 'density', 'onehot', 'argmax', None}:
        raise ValueError('`kind` must be in {counts, density, onehot, argmax, None}')
    # Let frame_step default to one full frame_length
    frame_step = frame_length if frame_step is None else frame_step
    # Process labels with a sliding window. TODO: vectorize?
    output = []
    for i in range(0, len(labels), frame_step):
        chunk = labels[i:i+frame_length]
        # Ignore incomplete end chunk unless padding is enabled
        if len(chunk) < frame_length and not pad_end:
            continue
        # Just append the chunk if kind is None
        if kind == None:
            output.append(chunk)
            continue
        # Count the occurences of each label
        counts = np.bincount(chunk, minlength=num_labels)
        # Then process based on kind
        if kind == 'counts':
            output.append(counts)
        elif kind == 'density':
            output.append(counts / len(chunk))
        elif kind == 'onehot':
            one_hot = np.zeros(num_labels)
            one_hot[np.argmax(counts)] = 1
            output.append(one_hot)
        elif kind == 'argmax':
            output.append(np.argmax(counts))
    if pad_end:
        return output
    else:
        return np.array(output)


EPS=1e-10
def normalize(x, mean, std):
    '''Normalizes the given tensor with Standard scaler'''
    return (x - mean) / std
