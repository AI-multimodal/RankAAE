#!/usr/bin/env python

import argparse
from rank_aae.post_hoc_explanation.latent2angularpdf.lat2ang_trainer import Latent2AngularPDFTrainer
import os
import yaml
import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_file', type=str, required=True,
                        help='File name of the dataset in Pickle format')
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='Config for training parameter in YAML format')
    parser.add_argument('-e', '--max_epoch', type=int, default=2000,
                        help='Maximum iterations')
    parser.add_argument('-v', '--verbose', action="store_true",
                        help='Maximum iterations')
    parser.add_argument('-w', "--work_dir", type=str, default='.',
                        help="Working directory to write the output files")
    parser.add_argument('-g', '--gpu_i', type=int, default=-1,
                        help='ID for GPU to use')
    args = parser.parse_args()

    work_dir = os.path.expandvars(os.path.expanduser(args.work_dir))
    with open(os.path.expandvars(os.path.expanduser(args.config))) as f:
        trainer_config = yaml.full_load(f)

    if not os.path.exists(work_dir):
        os.makedirs(work_dir, exist_ok=True)

    trainer = Latent2AngularPDFTrainer.from_data(os.path.expandvars(os.path.expanduser(args.data_file)),
        igpu=args.gpu_i, max_epoch=args.max_epoch, verbose=args.verbose, work_dir=work_dir,
        **trainer_config)

    t1 = datetime.datetime.now()
    metrics = trainer.train()
    t2 = datetime.datetime.now()
    dt = t2 - t1
    print(f"Training started at {t1.ctime()}, finished at {t2.ctime()}")
    print(f"In total used {dt.total_seconds()} seconds")
    
    print(metrics)
    trainer.test_models(work_dir=work_dir)


if __name__ == '__main__':
    main()
