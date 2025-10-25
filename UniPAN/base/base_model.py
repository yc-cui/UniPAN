from matplotlib.patheffects import Stroke, Normal
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import os
import torchmetrics.functional.image as MF
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import pandas as pd
import time
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from UniPAN.metric import D_lambda_torch, D_s_torch
from UniPAN.metric.d_rho import d_rho
from UniPAN.model.PanNet.network import PanNetModel
from UniPAN.util.misc import check_and_make, regularize_inputs
from UniPAN.loss import l1_loss
from sorcery import dict_of
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm
from io import BytesIO
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import seaborn as sns
plt.rcParams['mathtext.fontset'] = 'cm'
sns.set(style="ticks", palette="bright")


class BaseModel(pl.LightningModule):
    def __init__(self,
                 lr,
                 epochs,
                 bands,
                 rgb_c,
                 sensor,
                 scaler_type,
                 viz_dir
                 ):
        super().__init__()
        self.automatic_optimization = False
        self.viz_dir = viz_dir

        self.rgb_c = rgb_c
        self.sensor = sensor
        self.train_sensor = sensor
        self.scaler_type = scaler_type
        self.model = PanNetModel(bands)
        self.loss = l1_loss
        self.bands = bands

        self.reset_metrics()
        self.save_hyperparameters()

        self.visual_idx = [i for i in range(20)]

    def configure_optimizers(self):
        lr = self.hparams.lr
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        sche_opt = StepLR(opt, step_size=100, gamma=0.8)
        return [opt], [sche_opt]

    def forward(self, transformed_ms_down, transformed_pan_down, 
                ori_ms_down, ori_ms_down_up, ori_pan_down):
        raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        
        transformed_ms_down, transformed_pan_down, ori_ms_down, ori_ms_down_up, ori_pan_down = batch['transformed_ms_down'], batch['transformed_pan_down'], batch['ori_ms_down'], batch['ori_ms_down_up'], batch['ori_pan_down']
        gt = batch['gt']
        out = self.forward(transformed_ms_down, transformed_pan_down, ori_ms_down, ori_ms_down_up, ori_pan_down)
        pred = out["pred"]
        opt = self.optimizers()

        opt.zero_grad()
        total_loss, log_dict = self.loss(gt, pred)
        self.manual_backward(total_loss)
        opt.step()

        log_dict["lr"] = opt.param_groups[0]["lr"]
        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True)

    def on_train_epoch_end(self):
        sche_pf = self.lr_schedulers()
        sche_pf.step()

    def validation_step(self, batch, batch_idx):
        transformed_ms_down, transformed_pan_down, ori_ms_down, ori_ms_down_up, ori_pan_down = batch['transformed_ms_down'], batch['transformed_pan_down'], batch['ori_ms_down'], batch['ori_ms_down_up'], batch['ori_pan_down']
        
        gt = batch['gt']
        out = self.forward(transformed_ms_down, transformed_pan_down, ori_ms_down, ori_ms_down_up, ori_pan_down)
        pred = out["pred"]
        pred, gt, up_ms, ms, pan = regularize_inputs(pred, gt, ori_ms_down_up, transformed_ms_down, transformed_pan_down)
        self.save_full_ref(pred, gt)
  
        if batch_idx in self.visual_idx:
            channel_indices = torch.tensor(self.rgb_c, device=self.device)
            up_ms_rgb = torch.index_select(up_ms, 1, channel_indices)
            ms_rgb = torch.index_select(ms, 1, channel_indices)
            ms_rgb = F.interpolate(ms_rgb, scale_factor=4)
            pan_rgb = pan.repeat(1, 3, 1, 1)
            gt_rgb = torch.index_select(gt, 1, channel_indices)
            pred_rgb = torch.index_select(pred, 1, channel_indices)
            err_rgb = torch.abs(pred - gt).mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
            err_rgb /= torch.max(err_rgb)
            rgb_imgs = torch.cat([
                ms_rgb,
                pan_rgb,
                up_ms_rgb,
                pred_rgb,
                gt_rgb,
                err_rgb], dim=0)

            if self.visual is None:
                self.visual = rgb_imgs
            else:
                self.visual = torch.cat([self.visual, rgb_imgs], dim=0)

    def on_validation_epoch_end(self):
        model_name = self.__class__.__name__
        eval_results = {"method": model_name}
        for metric in self.eval_metrics:
            mean = np.nanmean(self.metrics_all[metric])
            std = np.nanstd(self.metrics_all[metric])
            eval_results[f'{metric}_mean'] = round(mean, 10)
            eval_results[f'{metric}_std'] = round(std, 10)
        filtered_dict = {k: v for k, v in eval_results.items() if isinstance(v, np.float64) and np.isnan(v) == False}
        self.log_dict(filtered_dict)
        filtered_dict["epoch"] = self.current_epoch
        csv_path = os.path.join(self.logger.save_dir, "metrics.csv")
        pd.DataFrame.from_dict(
            [filtered_dict]).to_csv(
            csv_path,
            mode="a",
            index=False,
            header=False if os.path.exists(csv_path) else True)

        grid = make_grid(self.visual, nrow=6, padding=2, normalize=False, scale_each=False, pad_value=0)
        image_grid = grid.permute(1, 2, 0).cpu().numpy()
        check_and_make(f"visual-{model_name}")
        save_path = f"visual-{model_name}/{self.current_epoch}.jpg"
        plt.imsave(save_path, image_grid)
        # self.logger.log_image(key="visual", images=[save_path])
        self.reset_metrics()

    def test_step(self, batch, batch_idx):
        transformed_ms_down, transformed_pan_down, ori_ms_down, ori_ms_down_up, ori_pan_down = batch['transformed_ms_down'], batch['transformed_pan_down'], batch['ori_ms_down'], batch['ori_ms_down_up'], batch['ori_pan_down']
        out = self.forward(transformed_ms_down, transformed_pan_down, ori_ms_down, ori_ms_down_up, ori_pan_down)       
        pred = out["pred"]
        pred, up_ms = regularize_inputs(pred, ori_ms_down_up)
        self.save_no_ref(ori_ms_down, ori_pan_down, pred, "test")
        self.save_RGB(pred, up_ms, batch_idx, self.rgb_c, self.sensor, "full")
        # self.save_GT(ori_pan_down, ori_ms_down_up, None, batch_idx, self.rgb_c, self.sensor, "full")


    def on_test_epoch_start(self):
        self.reset_metrics("test")

    def on_test_epoch_end(self):
        model_name = self.__class__.__name__
        eval_results = {"method": model_name}
        for metric in self.eval_metrics:
            mean = np.nanmean(self.metrics_all[metric])
            std = np.nanstd(self.metrics_all[metric])
            eval_results[f'{metric}_mean'] = round(mean, 10)
            eval_results[f'{metric}_std'] = round(std, 10)
        # csv_all_path = os.path.join(self.logger.save_dir, f"{self.sensor}_all.csv")
        # for k in self.metrics_all.keys():
        #     print(k, len(self.metrics_all[k]))
        # pd.DataFrame.from_dict(
        #     self.metrics_all).to_csv(
        #     csv_all_path,
        #     index=True,
        #     header=True)
        filtered_dict = {k: v for k, v in eval_results.items() if isinstance(
            v, np.float64) and np.isnan(v) == False and "std" not in k}
        print(filtered_dict)
        filtered_dict["epoch"] = "testing"
        # os.makedirs(os.path.join(self.logger.save_dir, self.sensor), exist_ok=True)
        csv_path = os.path.join(self.logger.save_dir, f"{self.sensor}.csv")
        pd.DataFrame.from_dict(
            [filtered_dict]).to_csv(
            csv_path,
            mode="a",
            index=False,
            header=False if os.path.exists(csv_path) else True)
        self.reset_metrics()

    def save_GT(self, pan, up_ms, gt, idx, RGB, sensor, type):
        def _linear_scale(gray, mi=0, ma=255):
            min_value = np.percentile(gray, 2)
            max_value = np.percentile(gray, 98)
            truncated_gray = np.clip(gray, a_min=min_value, a_max=max_value)
            processed_gray = ((truncated_gray - min_value) / (max_value - min_value)) * (ma - mi)
            return processed_gray

        def _normlize_uint8(img):
            # max, min = np.max(img, axis=(0, 1)), np.min(img, axis=(0, 1))
            # img = np.float32(img - min) / (max - min)
            img = np.clip(img, 0, 1)
            img = (img * 255).astype(np.uint8)
            return img

        def _get_RGB(img, RGB):
            pred_rgb = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            pred_rgb[:, :, 0] = _linear_scale(img[:, :, RGB[0]]).astype(np.uint8)
            pred_rgb[:, :, 1] = _linear_scale(img[:, :, RGB[1]]).astype(np.uint8)
            pred_rgb[:, :, 2] = _linear_scale(img[:, :, RGB[2]]).astype(np.uint8)
            return pred_rgb

        up_ms = up_ms.squeeze().permute(1, 2, 0).clone().cpu().numpy()
        pan = pan.squeeze().clone().cpu().numpy()

        upms_dir = os.path.join("visual_RGB-GT", sensor, type, "upms")
        pan_dir = os.path.join("visual_RGB-GT", sensor, type, "pan")
        os.makedirs(upms_dir, exist_ok=True)
        os.makedirs(pan_dir, exist_ok=True)
        plt.imsave(os.path.join(pan_dir, str(idx) + ".png"), _normlize_uint8(pan), cmap="gray")
        plt.imsave(os.path.join(upms_dir, str(idx) + ".png"), _get_RGB(up_ms, RGB))

        if gt is not None:
            gt = gt.squeeze().permute(1, 2, 0).clone().cpu().numpy()
            gt_dir = os.path.join("visual_RGB-GT", sensor, type, "gt")
            os.makedirs(gt_dir, exist_ok=True)
            plt.imsave(os.path.join(gt_dir, str(idx) + ".png"), _get_RGB(gt, RGB))

    def save_full_ref(self, pred, gt, split="val"):
        data_range = (0., 1.)
        self.record_metrics('PSNR', MF.peak_signal_noise_ratio(pred, gt, data_range=data_range), split)
        
    def save_RGB(self, pred, gt, idx, RGB, sensor, type):
        # return

        def _linear_scale(gray, mi=0, ma=255):
            min_value = np.percentile(gray, 2)
            max_value = np.percentile(gray, 98)
            truncated_gray = np.clip(gray, a_min=min_value, a_max=max_value)
            processed_gray = ((truncated_gray - min_value) / (max_value - min_value)) * (ma - mi)
            return processed_gray

        def _normlize_uint8(img):
            # max, min = np.max(img, axis=(0, 1)), np.min(img, axis=(0, 1))
            # img = np.float32(img - min) / (max - min)
            img = np.clip(img, 0, 1)
            img = (img * 255).astype(np.uint8)
            return img

        pred = pred.squeeze().permute(1, 2, 0).clone().cpu().numpy()
        pred[np.isnan(pred)] = 0.0
        gt = gt.squeeze().permute(1, 2, 0).clone().cpu().numpy()
        pred_rgb = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        pred_rgb[:, :, 0] = _linear_scale(pred[:, :, RGB[0]]).astype(np.uint8)
        pred_rgb[:, :, 1] = _linear_scale(pred[:, :, RGB[1]]).astype(np.uint8)
        pred_rgb[:, :, 2] = _linear_scale(pred[:, :, RGB[2]]).astype(np.uint8)
        pred_dir = os.path.join(f"viz/{self.viz_dir}", sensor, type, "pred")
        os.makedirs(pred_dir, exist_ok=True)
        plt.imsave(os.path.join(pred_dir, str(idx) + "_" + self.__class__.__name__ + ".png"), pred_rgb)



    def save_no_ref(self, lrms, pan, pred, split="val"):
        d_lambda = D_lambda_torch(lrms, pred)
        d_s = D_s_torch(lrms, pan, pred)
        qnr = (1 - d_lambda) * (1 - d_s)
        self.record_metrics('D_lambda', d_lambda, split)
        self.record_metrics('D_s', d_s, split)
        self.record_metrics('QNR', qnr, split)
        drfo = d_rho(F.interpolate(pred, scale_factor=0.25, mode="bicubic", align_corners=True), lrms)
        self.record_metrics('D_Rho', drfo, split)

    def reset_metrics(self, split="val"):
        self.eval_metrics = ['D_lambda', 'D_s', 'QNR', 'D_Rho', 'PSNR']
        self.eval_metrics = [f"{split}/" + i for i in self.eval_metrics]
        tmp_results = {}
        for metric in self.eval_metrics:
            tmp_results.setdefault(metric, [])

        self.metrics_all = tmp_results
        self.visual = None

    def record_metrics(self, k, v, split="val"):
        # if torch.isfinite(v):
            self.metrics_all[f'{split}/' + k].append(v.item())
