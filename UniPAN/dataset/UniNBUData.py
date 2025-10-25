
import pytorch_lightning as pl
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d
from torch.utils.data import DataLoader
from sorcery import dict_of
import os
from scipy import io
import random
import numpy as np
import scipy.stats as stats
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.utils import check_random_state, resample
from sklearn.utils.validation import (
    check_is_fitted,
    check_array,
    _check_feature_names_in,
    FLOAT_DTYPES,
)
from sklearn.utils._param_validation import Interval, Integral, StrOptions
from scipy import sparse
import warnings
from sklearn.preprocessing import QuantileTransformer
from UniPAN.dataset.sequence import seq

BOUNDS_THRESHOLD = 1e-7


def _blur_down(img, scale=0.25, ksize=(3, 3), sigma=(1.5, 1.5)):
    blur = gaussian_blur2d(img, ksize, sigma)
    return F.interpolate(blur, scale_factor=scale, mode="bicubic", align_corners=True)



class DatasetMinMaxScaler(Dataset):
    def __init__(self, data_dir, split="train", ori_test=False, seed=42, ms_transformer=None, pan_transformer=None, kwargs=None):
        self.data_dir = data_dir
        self.data_dir = os.path.expanduser(data_dir)
        self.split = split
        self.ori_test = ori_test
        random_seed = seed
        train_ratio = 0.7
        val_ratio = 0.1
        self.kwargs = kwargs

        mat_dir = os.path.join(self.data_dir, "MS_256")
        mat_files = sorted(os.listdir(mat_dir))
        num_files = len(mat_files)
        # print(mat_files)

        random.seed(random_seed)
        np.random.seed(random_seed)
        # random_sequence = list(range(num_files))
        # random.shuffle(random_sequence)

        if 'ikonos' in data_dir.lower():
            random_sequence = seq['ik']
        elif 'quickbird' in data_dir.lower():
            random_sequence = seq['qb']
        elif 'gaofen-1' in data_dir.lower():
            random_sequence = seq['gf1']
        elif 'worldview-2' in data_dir.lower():
            random_sequence = seq['wv2']
        elif 'worldview-3' in data_dir.lower():
            random_sequence = seq['wv3']
        elif 'worldview-4' in data_dir.lower():
            random_sequence = seq['wv4']
        else:
            print('Found no sequence!')
            exit(0)

        train_idx = int(num_files * train_ratio)
        val_idx = train_idx + int(num_files * val_ratio)

        self.train_mat_files_files = [mat_files[idx] for idx in random_sequence[:train_idx]]
        self.val_mat_files_files = [mat_files[idx] for idx in random_sequence[train_idx:val_idx]]
        self.test_mat_files_files = [mat_files[idx] for idx in random_sequence[val_idx:]]

        # print(data_dir, ":", self.split, ":")
        # print("train:")
        # print(self.train_mat_files_files)
        # print("val:")
        # print(self.val_mat_files_files)
        # print("test:")
        # print(self.test_mat_files_files)

        if self.split == "train":
            self.mat_files = self.train_mat_files_files
        elif self.split == "test":
            self.mat_files = self.test_mat_files_files
        elif self.split == "val":
            self.mat_files = self.val_mat_files_files
        else:
            raise RuntimeError("Wrong split.")

        print(data_dir, ":", self.split, ":", self.mat_files[:10])

        if "gaofen" in data_dir.lower():
            # print("gaofen")
            self.max_val = 1023
        else:
            self.max_val = 2047

        self.ms_transformer = ms_transformer
        self.pan_transformer = pan_transformer
        # self.crop = RandomCrop((64, 64), p=1., cropping_mode="resample")

        if split == "train":
            self.fit()
        # else:
        #     tmp = self.mat_files
        #     self.mat_files = self.train_mat_files_files
        #     self.fit()
        #     self.mat_files = tmp

    def __len__(self):
        return len(self.mat_files)

    def __getitem__(self, idx):

        mat_ms = io.loadmat(os.path.join(self.data_dir, "MS_256", self.mat_files[idx]))
        mat_pan = io.loadmat(os.path.join(self.data_dir, "PAN_1024", self.mat_files[idx]))

        # Some keys may be different.
        key_ms = "imgMS" if "imgMS" in mat_ms.keys() else "I_MS"
        key_pan = "imgPAN" if "imgPAN" in mat_pan.keys() else "I_PAN"
        if "imgPAN" not in mat_pan.keys() and "I_PAN" not in mat_pan.keys():
            key_pan = "block"
        ms = torch.from_numpy((mat_ms[key_ms] / self.max_val).astype(np.float32)).permute(2, 0, 1) # chw
        if ms.shape[0] == 8:
            ms = ms[[1, 2, 4, 6], ...]
        pan = torch.from_numpy((mat_pan[key_pan] / self.max_val).astype(np.float32)).unsqueeze(0) # 1hw

        ori_ms = ms
        gt = ms
        ori_pan = pan

        if self.split == "train" or self.split == "val" or not self.ori_test:
            ori_ms_down = _blur_down(ori_ms.unsqueeze(0)).squeeze(0)
            ori_pan_down = _blur_down(ori_pan.unsqueeze(0)).squeeze(0)
        else:
            ori_ms_down = ori_ms
            ori_pan_down = ori_pan

        if self.ms_transformer is not None:
            transformed_ms_down = self.transform_ms(ori_ms_down).float()
        else:
            transformed_ms_down = ori_ms_down
        if self.pan_transformer is not None:
            transformed_pan_down = self.transform_pan(ori_pan_down).float()
        else:
            transformed_pan_down = ori_pan_down

        ori_ms_down_up = F.interpolate(ori_ms_down.unsqueeze(0), (ori_pan_down.shape[-2], ori_pan_down.shape[-1]), mode="bicubic", align_corners=True).squeeze(0)

        inp_dict = dict_of(transformed_ms_down, 
                           transformed_pan_down, 
                           ori_ms_down,
                           ori_ms_down_up, 
                           ori_pan_down,
                           gt
                           )

        return inp_dict

    def fit(self):
        pass

    def transform_ms(self, ms):
        pass

    def transform_pan(self, pan):
        pass

    def inv_transform_ms(self, ms):
        if self.ms_transformer is None:
            return ms
        ms = ms.squeeze(0)
        C, H, W = ms.shape
        ms = self.ms_transformer.inverse_transform(ms.reshape(C, -1).transpose(1, 0)).reshape(H, W, C).transpose(2, 0, 1)
        return torch.from_numpy(ms).unsqueeze(0).float()

    def inv_transform_pan(self, pan):
        if self.pan_transformer is None:
            return pan
        pan = pan.squeeze(0)
        C, H, W = pan.shape
        pan = self.pan_transformer.transform(pan.reshape(C, -1).transpose(1, 0)).reshape(H, W, C).transpose(2, 0, 1)
        return torch.from_numpy(pan).unsqueeze(0).float()






class UniDistribution(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Transform features using quantiles information.
    
    Extended to support custom target distributions through target_samples parameter.
    """

    _parameter_constraints: dict = {
        "n_quantiles": [Interval(Integral, 1, None, closed="left")],
        "output_distribution": [StrOptions({"uniform", "normal", "custom"})],
        "target_samples": [np.ndarray, None],
        "ignore_implicit_zeros": ["boolean"],
        "subsample": [Interval(Integral, 1, None, closed="left"), None],
        "random_state": ["random_state"],
        "copy": ["boolean"],
    }

    def __init__(
        self,
        *,
        n_quantiles=1000,
        output_distribution="uniform",
        target_samples=None,
        ignore_implicit_zeros=False,
        subsample=10_000,
        random_state=None,
        copy=True,
    ):
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.target_samples = target_samples
        self.ignore_implicit_zeros = ignore_implicit_zeros
        self.subsample = subsample
        self.random_state = random_state
        self.copy = copy

    def _dense_fit(self, X, random_state):
        if self.ignore_implicit_zeros:
            warnings.warn(
                "'ignore_implicit_zeros' only applies to sparse matrices", UserWarning
            )

        n_samples, _ = X.shape
        if self.subsample is not None and n_samples > self.subsample:
            X = resample(
                X,
                replace=False,
                n_samples=self.subsample,
                random_state=random_state,
            )

        self.quantiles_ = np.nanpercentile(X, self.references_ * 100, axis=0)
        self.quantiles_ = np.maximum.accumulate(self.quantiles_)

    def _sparse_fit(self, X, random_state):
        n_samples, n_features = X.shape
        X = X.tocsc()
        self.quantiles_ = []
        for col in range(n_features):
            col_data = X.data[X.indptr[col] : X.indptr[col + 1]]
            if self.subsample is not None and len(col_data) > self.subsample:
                col_data = random_state.choice(
                    col_data, self.subsample, replace=False
                )
            if self.ignore_implicit_zeros:
                col_data = np.append(col_data, [0.0] * (n_samples - len(col_data)))
            self.quantiles_.append(np.nanpercentile(col_data, self.references_ * 100))
        self.quantiles_ = np.array(self.quantiles_).T
        self.quantiles_ = np.maximum.accumulate(self.quantiles_)

    def _prepare_target_quantiles(self):
        if self.output_distribution == "custom":
            if self.target_samples is None:
                raise ValueError(
                    "target_samples must be provided for custom distribution"
                )
            target = check_array(
                self.target_samples,
                ensure_2d=False,
                dtype=FLOAT_DTYPES,
                force_all_finite="allow-nan",
            )
            self.target_quantiles_ = np.nanpercentile(target, self.references_ * 100)
            self.target_quantiles_ = np.maximum.accumulate(self.target_quantiles_)

    # @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        X = self._validate_data(
            X,
            accept_sparse="csc",
            dtype=FLOAT_DTYPES,
            force_all_finite="allow-nan",
        )
        n_samples = X.shape[0]
        self.n_quantiles_ = min(max(self.n_quantiles, 1), n_samples)
        self.references_ = np.linspace(0, 1, self.n_quantiles_, endpoint=True)
        self._prepare_target_quantiles()

        rng = check_random_state(self.random_state)
        if sparse.issparse(X):
            self._sparse_fit(X, rng)
        else:
            self._dense_fit(X, rng)
        return self

    def _transform_col(self, X_col, quantiles, inverse):
        if inverse:
            return self._inverse_transform_col(X_col, quantiles)
        
        # Forward transform logic
        X_col = np.interp(X_col, quantiles, self.references_)
        
        # Apply target distribution mapping
        if self.output_distribution == "normal":
            X_col = self._normal_ppf(X_col)
        elif self.output_distribution == "custom":
            X_col = np.interp(X_col, self.references_, self.target_quantiles_)
        
        return X_col

    def _inverse_transform_col(self, X_col, quantiles):
        if self.output_distribution == "normal":
            X_col = stats.norm.cdf(X_col)
        elif self.output_distribution == "custom":
            X_col = np.interp(X_col, self.target_quantiles_, self.references_)
        
        return np.interp(X_col, self.references_, quantiles)

    def _normal_ppf(self, X_col):
        X_col = stats.norm.ppf(np.clip(X_col, BOUNDS_THRESHOLD, 1 - BOUNDS_THRESHOLD))
        return np.clip(X_col, stats.norm.ppf(BOUNDS_THRESHOLD), 
                       stats.norm.ppf(1 - BOUNDS_THRESHOLD))

    def transform(self, X):
        check_is_fitted(self)
        X = self._validate_data(
            X,
            reset=False,
            accept_sparse="csc",
            copy=self.copy,
            dtype=FLOAT_DTYPES,
            force_all_finite="allow-nan",
        )
        self._check_feature_count(X)
        
        if sparse.issparse(X):
            for i in range(X.shape[1]):
                sl = slice(X.indptr[i], X.indptr[i + 1])
                X.data[sl] = self._transform_col(X.data[sl], self.quantiles_[:, i], False)
        else:
            for i in range(X.shape[1]):
                X[:, i] = self._transform_col(X[:, i], self.quantiles_[:, i], False)
        return X

    def inverse_transform(self, X):
        check_is_fitted(self)
        X = self._validate_data(
            X,
            reset=False,
            accept_sparse="csc",
            copy=self.copy,
            dtype=FLOAT_DTYPES,
            force_all_finite="allow-nan",
        )
        self._check_feature_count(X)
        
        if sparse.issparse(X):
            for i in range(X.shape[1]):
                sl = slice(X.indptr[i], X.indptr[i + 1])
                X.data[sl] = self._transform_col(X.data[sl], self.quantiles_[:, i], True)
        else:
            for i in range(X.shape[1]):
                X[:, i] = self._transform_col(X[:, i], self.quantiles_[:, i], True)
        return X

    def _check_feature_count(self, X):
        if X.shape[1] != self.quantiles_.shape[1]:
            raise ValueError(f"X has {X.shape[1]} features but expected {self.quantiles_.shape[1]}")

    def _more_tags(self):
        return {"allow_nan": True, "preserves_dtype": [np.float64]}



class DatasetQuantileScaler(DatasetMinMaxScaler):
    def __init__(self, data_dir, split="train", ori_test=False, seed=42, ms_transformer=None, pan_transformer=None, kwargs=None):
        super().__init__(data_dir, split, ori_test, seed, ms_transformer, pan_transformer, kwargs)

    def __len__(self):
        return len(self.mat_files)

    def fit(self):
        data_ms = []
        data_pan = []
        for i in range(len(self)):
            ms = self[i]['ori_ms_down']
            pan = self[i]['ori_pan_down']
            if ms.shape[0] == 8:
                ms = ms[[1, 2, 4, 6], ...]
            data_ms.append(ms.reshape(4, -1))
            data_pan.append(pan.reshape(-1))
        data_ms = np.array(data_ms).transpose(0, 2, 1).reshape(-1, 4)
        data_pan = np.array(data_pan).reshape(-1, 1)
        if self.kwargs['out_dist'] == 'custom':
            self.ms_transformer = UniDistribution(
                output_distribution=self.kwargs['out_dist'],
                target_samples=self.kwargs['target_samples'],
                n_quantiles=self.kwargs['n_quantiles'],
                subsample=self.kwargs['subsample'],
                random_state=42)
            self.pan_transformer = UniDistribution(
                output_distribution=self.kwargs['out_dist'],
                target_samples=self.kwargs['target_samples'],
                n_quantiles=self.kwargs['n_quantiles'],
                subsample=self.kwargs['subsample'],
                random_state=42)
        else:
            self.ms_transformer = QuantileTransformer(
                output_distribution=self.kwargs['out_dist'],
                n_quantiles=self.kwargs['n_quantiles'],
                subsample=self.kwargs['subsample'],
                random_state=42)
            self.pan_transformer = QuantileTransformer(
                output_distribution=self.kwargs['out_dist'],
                n_quantiles=self.kwargs['n_quantiles'],
                subsample=self.kwargs['subsample'],
                random_state=42)
        self.ms_transformer.fit(data_ms)
        self.pan_transformer.fit(data_pan)

    def transform_ms(self, ms):
        if not self.kwargs["trans_ms"]:
            return ms
        C, H, W = ms.shape
        ms = self.ms_transformer.transform(ms.reshape(C, -1).transpose(1, 0)).reshape(H, W, C).transpose(2, 0, 1)
        return torch.from_numpy(ms) 

    def transform_pan(self, pan):
        if not self.kwargs["trans_pan"]:
            return pan
        C, H, W = pan.shape
        pan = self.pan_transformer.transform(pan.reshape(C, -1).transpose(1, 0)).reshape(H, W, C).transpose(2, 0, 1)
        return torch.from_numpy(pan)




class plNBUDataset(pl.LightningDataModule):
    def __init__(self, scaler, data_dir_train, data_dirs_test, batch_size, num_workers=4, pin_memory=True, seed=42, kwargs=None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.dataset_train = scaler(data_dir_train, split="train", seed=seed, kwargs=kwargs)
        self.dataset_val = scaler(
            data_dir_train,
            split="val",
            seed=seed,
            ms_transformer=self.dataset_train.ms_transformer,
            pan_transformer=self.dataset_train.pan_transformer, kwargs=kwargs)

        self.dataset_test_ori = scaler(
            data_dirs_test,
            split="test",
            ori_test=True,
            seed=seed,
            ms_transformer=self.dataset_train.ms_transformer,
            pan_transformer=self.dataset_train.pan_transformer, kwargs=kwargs)

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test_ori,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
