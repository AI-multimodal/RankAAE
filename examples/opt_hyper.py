#!/usr/bin/env python

import argparse
import socket
from contextlib import contextmanager
import multiprocessing
from optuna.trial import Trial
from sc.clustering.trainer import Trainer
import optuna
import os
import yaml
import time
import numpy as np
from optuna.pruners import HyperbandPruner
import subprocess


class TrainerCallBack:
    def __init__(self, merge_objectives, trial: Trial):
        super().__init__()
        self.merge_objectives = merge_objectives
        self.trial = trial

    def __call__(self, epoch, metrics):
        if self.merge_objectives:
            weights = [100, 50.0, 50.0, -1.0, -1.0, -1.0, -1.0]
            metrics = (np.array(weights) * np.array(metrics)).sum()
        else:
            metrics = metrics[0]

        self.trial.report(metrics, epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()


class GpuQueue:
    def __init__(self, n_gpus, n_jobs):
        self.queue = multiprocessing.Manager().Queue()
        all_idxs = list(range(n_jobs)) if n_gpus > 0 else [None]
        self.n_gpus = n_gpus
        for idx in all_idxs:
            self.queue.put(idx)

    @contextmanager
    def one_gpu_per_process(self):
        current_idx = self.queue.get()
        yield current_idx % self.n_gpus
        self.queue.put(current_idx)


class Objective:
    def __init__(self, gpu_queue: GpuQueue, trainer_args, opt_config, base_trail_number,
                 single_objective, merge_objectives):
        super().__init__()
        self.gpu_queue = gpu_queue
        self.trainer_args = trainer_args
        self.opt_config = opt_config
        self.base_trail_number = base_trail_number
        self.single_objective = single_objective
        self.merge_objectives = merge_objectives

    def __call__(self, trial: Trial):
        with self.gpu_queue.one_gpu_per_process() as gpu_i:
            kwargs = {}
            for k, v in self.opt_config.items():
                if v["sampling"] != 'the categorical':
                    low, high = v["low"], v["high"]
                if v["sampling"] == 'int':
                    kwargs[k] = trial.suggest_int(name=k, low=low, high=high)
                elif v["sampling"] == 'uniform':
                    kwargs[k] = trial.suggest_uniform(name=k, low=low, high=high)
                elif v["sampling"] == 'loguniform':
                    kwargs[k] = trial.suggest_loguniform(name=k, low=low, high=high)
                elif v["sampling"] == 'categorical':
                    kwargs[k] = trial.suggest_categorical(name=k, choices=v["choices"])
            work_dir = f'{os.path.expandvars(os.path.expanduser(self.trainer_args.work_dir))}/trials' \
                       f'/{self.base_trail_number+trial.number:05d}_{time.time_ns() - 1597090000000000000}'
            trainer = Trainer.from_data(self.trainer_args.data_file,
                                        igpu=gpu_i,
                                        max_epoch=self.trainer_args.max_epoch,
                                        verbose=self.trainer_args.verbose,
                                        work_dir=work_dir,
                                        **kwargs)
            if self.single_objective:
                trainer_callback = TrainerCallBack(self.merge_objectives, trial)
            else:
                trainer_callback = None
            metrics = trainer.train(trainer_callback)
            if self.single_objective:
                if self.merge_objectives:
                    weights = [100, 50.0, 50.0, -1.0, -1.0, -1.0, -1.0]
                    metrics = (np.array(weights) * np.array(metrics)).sum()
                else:
                    metrics = metrics[0]
        return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='Config for parameter to optimize in YAML format')
    parser.add_argument('-d', '--data_file', type=str, required=True,
                        help='File name of the dataset in CSV format')
    parser.add_argument('-e', '--max_epoch', type=int, default=2000,
                        help='Maximum iterations')
    parser.add_argument('-v', '--verbose', action="store_true",
                        help='Maximum iterations')
    parser.add_argument('-w', "--work_dir", type=str, default='.',
                        help="Working directory to write the output files")
    parser.add_argument('-g', '--gpus', type=int, default=4,
                        help='Number of GPUs')
    parser.add_argument('-j', '--jobs', type=int, default=8,
                        help='Number of Optuna trial jobs')
    parser.add_argument('-t', '--trials', type=int, default=50,
                        help='Number of total trails to evaluate model')
    parser.add_argument('--name', type=str, default='opt_daae',
                        help='Number of total trails to evaluate model')
    parser.add_argument('-s', "--single", action="store_true",
                        help='Optimize first metric only, this option will activate pruner')
    parser.add_argument('-m', "--merge_objectives", action="store_true",
                        help='Merge all all metrix into one')
    parser.add_argument("--min_resource", type=int, default=50,
                        help='Min Resource for HyperbandPruner')
    args = parser.parse_args()

    work_dir = os.path.expandvars(os.path.expanduser(args.work_dir))

    task_id = 0
    job_id = 0
    if 'SLURM_PROCID' in os.environ:
        task_id = int(os.environ['SLURM_PROCID'])
        job_id = int(os.environ['SLURM_JOB_ID'])
        sleep_seconds = task_id * 3
        print(f"Task ID is {task_id}, will sleep {sleep_seconds} seconds before start")
        time.sleep(sleep_seconds)

    hostname = socket.gethostname().split('.', 1)[0]
    if not os.path.exists(f'{work_dir}/resource_usage_{job_id}'):
        os.makedirs(f'{work_dir}/resource_usage_{job_id}', exist_ok=True)
    cpu_log_file = open(f'{work_dir}/resource_usage_{job_id}/cpu_{task_id}_{hostname}.txt', 'wt')
    subprocess.Popen(['/usr/bin/top', '-i', '-b', '-d', '31'],
                     stdout=cpu_log_file)
    gpu_log_file = open(f'{work_dir}/resource_usage_{job_id}/gpu_{task_id}_{hostname}.txt', 'wt')
    subprocess.Popen(['/usr/bin/nvidia-smi', '-l', '37'],
                     stdout=gpu_log_file)

    with open(args.config) as f:
        opt_config = yaml.full_load(f)

    if not os.path.exists(work_dir):
        os.makedirs(work_dir, exist_ok=True)
    single_objective = args.single
    merge_objectives = args.merge_objectives
    if single_objective:
        study = optuna.create_study(
            direction='maximize',
            study_name=args.name,
            storage=f'sqlite:///{work_dir}/{args.name}.db',
            load_if_exists=True,
            pruner=HyperbandPruner(min_resource=args.min_resource)
        )
    else:
        study = optuna.multi_objective.create_study(
            directions=['maximize'] * 3 + ["minimize"] * 4,
            study_name=args.name,
            storage=f'sqlite:///{work_dir}/{args.name}.db',
            load_if_exists=True)
    base_trail_number = len(study.trials)
    gq = GpuQueue(args.gpus, args.jobs)
    obj = Objective(gq, args, opt_config, base_trail_number, single_objective, merge_objectives)
    study.optimize(obj, n_trials=args.trials, n_jobs=args.jobs)

    print("Number of finished trials: ", len(study.trials))
    if single_objective:
        print(f"Best Trial#: {study.best_trial.number}")
        print(f"Best Value:  {study.best_value}")
        print(f"Best Params: {study.best_params}")
    else:
        print("Pareto front:")
        trials = {str(trial.values): trial for trial in study.get_pareto_front_trials()}
        trials = list(trials.values())
        trials.sort(key=lambda t: t.values)
        for trial in trials:
            print("  Trial#{}".format(trial.number))
            print("    Values: ".format(trial.values))
            print("    Params: {}".format(trial.params))
            print()


if __name__ == '__main__':
    main()
