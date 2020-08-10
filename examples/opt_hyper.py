import argparse
from contextlib import contextmanager
import multiprocessing
from optuna.trial import Trial
from sc.clustering.trainer import Trainer
import optuna
import os
import yaml
import time
import matplotlib.pyplot as plt


class GpuQueue:
    def __init__(self, n_gpus):
        self.queue = multiprocessing.Manager().Queue()
        all_idxs = list(range(n_gpus)) if n_gpus > 0 else [None]
        for idx in all_idxs:
            self.queue.put(idx)

    @contextmanager
    def one_gpu_per_process(self):
        current_idx = self.queue.get()
        yield current_idx
        self.queue.put(current_idx)


class Objective:
    def __init__(self, gpu_queue: GpuQueue, trainer_args, opt_config, base_trail_number):
        super().__init__()
        self.gpu_queue = gpu_queue
        self.trainer_args = trainer_args
        self.opt_config = opt_config
        self.base_trail_number = base_trail_number

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
            metrics = trainer.train()
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
    parser.add_argument('-j', '--jobs', type=int, default=4,
                        help='Number of Optuna trial jobs')
    parser.add_argument('-t', '--trials', type=int, default=50,
                        help='Number of total trails to evaluate model')
    parser.add_argument('--name', type=str, default='opt_daae',
                        help='Number of total trails to evaluate model')
    parser.add_argument('-s', "--single", action="store_true",
                        help='Optimize first metric only, this option will activate pruner')
    args = parser.parse_args()

    plt.switch_backend('Agg')

    with open(args.config) as f:
        opt_config = yaml.full_load(f)

    work_dir = os.path.expandvars(os.path.expanduser(args.work_dir))
    if not os.path.exists(work_dir):
        os.makedirs(work_dir, exist_ok=True)

    study = optuna.multi_objective.create_study(
        directions=["minimize"] * 7,
        study_name=args.name,
        storage=f'sqlite:///{work_dir}/{args.name}.db',
        load_if_exists=True,
        )
    base_trail_number = len(study.trials)
    gq = GpuQueue(args.gpus)
    obj = Objective(gq, args, opt_config, base_trail_number)
    study.optimize(obj, n_trials=args.trials, n_jobs=args.jobs)

    print("Number of finished trials: ", len(study.trials))
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
