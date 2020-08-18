#!/usr/bin/env python

import argparse
import os
import time

import numpy as np
import optuna
import yaml
from optuna.pruners import HyperbandPruner
from optuna.trial import Trial
from optuna.exceptions import OptunaError, TrialPruned

from sc.clustering.trainer import Trainer


class TrainerCallBack:
    def __init__(self, merge_objectives, trial: Trial):
        super().__init__()
        self.merge_objectives = merge_objectives
        self.trial = trial
        self.metric_weights = [1.0] * 3

    def __call__(self, epoch, metrics):
        if self.merge_objectives:
            metrics = (np.array(self.metric_weights) * np.array(metrics)).sum()
        else:
            metrics = metrics[0]

        self.trial.report(metrics, epoch)
        if self.trial.should_prune():
            raise TrialPruned()


class Objective:
    def __init__(self, igpu, trainer_args, opt_config, base_trail_number,
                 single_objective, merge_objectives):
        super().__init__()
        self.igpu = igpu
        self.trainer_args = trainer_args
        self.opt_config = opt_config
        self.base_trail_number = base_trail_number
        self.single_objective = single_objective
        self.merge_objectives = merge_objectives

    def __call__(self, trial: Trial, max_redo=5):
        kwargs = {}
        for k, v in self.opt_config.items():
            if v["sampling"] != 'the categorical':
                low, high = v["low"], v["high"]
            else:
                low, high = None, None
            if v["sampling"] == 'int':
                kwargs[k] = trial.suggest_int(name=k, low=low, high=high)
            elif v["sampling"] == 'uniform':
                kwargs[k] = trial.suggest_uniform(name=k, low=low, high=high)
            elif v["sampling"] == 'loguniform':
                kwargs[k] = trial.suggest_loguniform(name=k, low=low, high=high)
            elif v["sampling"] == 'categorical':
                kwargs[k] = trial.suggest_categorical(name=k, choices=v["choices"])
        if self.single_objective:
            trainer_callback = TrainerCallBack(self.merge_objectives, trial)
        else:
            trainer_callback = None
        metrics = 0.0
        for _ in range(max_redo):
            try:
                work_dir = f'{os.path.expandvars(os.path.expanduser(self.trainer_args.work_dir))}/trials' \
                           f'/{trial.number:05d}_{time.time_ns() - 1597090000000000000}'
                trainer = Trainer.from_data(self.trainer_args.data_file,
                                            igpu=self.igpu,
                                            max_epoch=self.trainer_args.max_epoch,
                                            verbose=self.trainer_args.verbose,
                                            work_dir=work_dir,
                                            **kwargs)
                metrics = trainer.train(trainer_callback)
                redo = False
            except OptunaError:
                raise
            except RuntimeError:
                redo = True
            if not redo:
                break
        else:
            print(f"Can't fix train error after tied {max_redo} times")
        if self.single_objective:
            if self.merge_objectives:
                metrics = (np.array(trainer_callback.metric_weights) * np.array(metrics)).sum()
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
    parser.add_argument('-g', '--gpu_i', type=int, default=0,
                        help='ID for GPU to use')
    parser.add_argument('-a', '--db_address', type=str, default='127.0.0.1',
                        help='hostname for the Redis DB sever')
    parser.add_argument('-p', '--db_port', type=int, default=6379,
                        help='Socket port for the Redis DB sever')
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
    parser.add_argument("--timeout", type=int, default=None,
                        help='Maximum time allowed per trial')
    args = parser.parse_args()

    work_dir = os.path.expandvars(os.path.expanduser(args.work_dir))

    with open(args.config) as f:
        opt_config = yaml.full_load(f)

    if not os.path.exists(work_dir):
        os.makedirs(work_dir, exist_ok=True)
    single_objective = args.single
    merge_objectives = args.merge_objectives
    storage = optuna.storages.RedisStorage(url=f'redis://{args.db_address}:{args.db_port}')

    if single_objective:
        study = optuna.create_study(
            direction='maximize',
            study_name=args.name,
            storage=storage,
            load_if_exists=True,
            pruner=HyperbandPruner(min_resource=args.min_resource)
        )
    else:
        study = optuna.multi_objective.create_study(
            directions=['maximize'] * 3 + ["minimize"] * 4,
            study_name=args.name,
            storage=storage,
            load_if_exists=True)
    base_trail_number = len(study.trials)
    obj = Objective(args.gpu_i, args, opt_config, base_trail_number, single_objective, merge_objectives)
    study.optimize(obj, n_trials=args.trials, timeout=args.timeout)

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
