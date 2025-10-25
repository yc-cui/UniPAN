
import argparse
import pytorch_lightning as pl
import os
import os, sys

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_path not in sys.path:
    sys.path.append(project_path)

from pytorch_lightning import seed_everything
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger, CSVLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from UniPAN.dataset.UniNBUData import  plNBUDataset
from UniPAN.dataset.UniNBUData import DatasetMinMaxScaler
from UniPAN.dataset.UniNBUData import DatasetQuantileScaler


from UniPAN.model import FeINFNetModel
from UniPAN.util.misc import check_and_make


sensor2dir = {
    'wv2': 'YOUR/DATA/DIR/5 WorldView-2',
    'gf1': 'YOUR/DATA/DIR/3 Gaofen-1',
    'ik': 'YOUR/DATA/DIR/1 IKONOS',
    'wv3': 'YOUR/DATA/DIR/6 WorldView-3',
    'wv4': 'YOUR/DATA/DIR/4 WorldView-4',
    'qb': 'YOUR/DATA/DIR/2 QuickBird',
}


scalers = {
    'spec': DatasetMinMaxScaler,
    'uni': DatasetQuantileScaler,
}


def get_args_parser():
    parser = argparse.ArgumentParser('UniPAN training', add_help=False)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--epochs', default=600, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)
    parser.add_argument('--ms_chans', default=4, type=int)
    parser.add_argument('--scaler', default='uni', type=str)
    parser.add_argument('--rgb_c', default='2,1,0')
    parser.add_argument('--train_sensor', default='wv2', type=str)
    parser.add_argument('--test_sensors', default='ik,gf1,wv3,wv4,qb', type=str)
    parser.add_argument('--test_freq', default=10, type=int)
    parser.add_argument('--device', default=1, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',)
    parser.add_argument('--min_lr', type=float, default=1e-4, metavar='LR')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--out_dist', default='uniform', choices=["uniform", "normal", "custom"])
    parser.add_argument('--n_quantiles', default=1000, type=int)
    parser.add_argument('--subsample', default=10000, type=int)

    return parser


def main(args):

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    model_name = "UniPan." + FeINFNetModel.__name__

    output_dir = f"logs/log_m={model_name}_s={args.train_sensor}_sd={args.seed}_sc={args.scaler}"
    
    if args.scaler == "uni":
        output_dir = f"logs_uni/log_m={model_name}_s={args.train_sensor}_sd={args.seed}_sc={args.scaler}_d={args.out_dist}_nq={args.n_quantiles}_nsub={args.subsample}"

    check_and_make(output_dir)
    seed_everything(args.seed)

    dir_train_sensor = sensor2dir[args.train_sensor]
    scaler = scalers[args.scaler]
    print(scaler)
    dataset = plNBUDataset(scaler,
                           dir_train_sensor,
                           dir_train_sensor,
                           args.batch_size,
                           args.num_workers,
                           args.pin_mem,
                           args.seed,
                           kwargs={
                               "out_dist": args.out_dist,
                               "n_quantiles": args.n_quantiles,
                               "subsample": args.subsample,}
                           )
    model = FeINFNetModel(min_lr=args.min_lr,
                    lr=args.lr,
                    warmup_epochs=args.warmup_epochs,
                    epochs=args.epochs,
                    bands=args.ms_chans,
                    rgb_c=[int(c) for c in args.rgb_c.split(",")],
                    sensor=args.train_sensor,
                    scaler_type=args.scaler,
                    viz_dir=output_dir
                    )

    if args.wandb:
        wandb_logger = WandbLogger(project=model_name, name=output_dir, save_dir=output_dir)
    else:
        wandb_logger = [CSVLogger(name=output_dir, save_dir=output_dir)]
        wandb_logger.append(TensorBoardLogger(name=output_dir, save_dir=output_dir))
        
    model_checkpoint = ModelCheckpoint(dirpath=output_dir,
                                       monitor='val/PSNR_mean',
                                       mode="max",
                                       save_top_k=1,
                                       auto_insert_metric_name=False,
                                       filename='ep={epoch}_PSNR={val/PSNR_mean:.4f}',
                                       every_n_epochs=args.test_freq
                                       )

    trainer = pl.Trainer(max_epochs=args.epochs,
                         accelerator="gpu",
                         devices=[args.device],
                         logger=wandb_logger,
                         check_val_every_n_epoch=args.test_freq,
                         callbacks=[model_checkpoint],
                        #  precision='16-mixed'
                         )

    if not args.test_only:
        trainer.fit(model, dataset)
        trainer.test(ckpt_path="best", datamodule=dataset)
    else:
        trainer.test(model, ckpt_path=args.ckpt, datamodule=dataset)

    sensors = args.test_sensors.split(",")
    dir_test_sensors = [sensor2dir[s] for s in sensors]
    for sensor, dir_test_sensor in zip(sensors, dir_test_sensors):
        dataset = plNBUDataset(scaler,
                               dir_train_sensor,
                               dir_test_sensor,
                               args.batch_size,
                               args.num_workers,
                               args.pin_mem,
                               args.seed,
                               kwargs={
                                    "out_dist": args.out_dist,
                                    "n_quantiles": args.n_quantiles,
                                    "subsample": args.subsample,
               }
                               )
        model.sensor = sensor
        if not args.test_only:
            trainer.test(model, ckpt_path="best", datamodule=dataset)
        else:
            trainer.test(model, ckpt_path=args.ckpt, datamodule=dataset)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)

