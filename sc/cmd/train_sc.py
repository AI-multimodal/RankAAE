#!/usr/bin/env python

import argparse

import torch
from sc.clustering.trainer import Trainer
from sc.utils.parameter import Parameters
from sc.utils.logger import create_logger
import os
import yaml
import socket
import ipyparallel as ipp
import logging
import signal
import time
import numpy as np


engine_id = -1

def timeout_handler(signum, frame):
    raise Exception("Training Overtime!")


def get_parallel_map_func(work_dir=".", logger=logging.getLogger("Parallel")):
    
    c = ipp.Client(
        connection_info=f"{work_dir}/ipypar/security/ipcontroller-client.json"
    )

    with c[:].sync_imports():
        import torch
        from sc.clustering.trainer import Trainer
        from sc.utils.parameter import Parameters
        from sc.utils.logger import create_logger
        import os
        import socket
        import logging
        import signal
        import time
    logger.info(f"Engine IDs: {c.ids}")
    c[:].push(dict(run_training=run_training, timeout_handler=timeout_handler),
              block=True)

    return c.load_balanced_view().map_sync, len(c.ids)


def run_training(
    job_number, 
    work_dir, 
    trainer_config, 
    max_epoch, 
    verbose, 
    data_file, 
    timeout_hours=0,
    logger = logging.getLogger("training")
):

    work_dir = f'{work_dir}/training/job_{job_number+1}'
    if not os.path.exists(work_dir):
        os.makedirs(work_dir, exist_ok=True)

    logger = create_logger(f"subtraining_{job_number+1}", os.path.join(work_dir, "messages.txt"))

    if torch.get_num_interop_threads() > 2:
        torch.set_num_interop_threads(1)
        torch.set_num_threads(1)
    
    ngpus_per_node = torch.cuda.device_count()
    if "SLURM_LOCALID" in os.environ:
        local_id = int(os.environ.get("SLURM_LOCALID", 0))
    else:
        local_id = 0
    igpu = local_id % ngpus_per_node if torch.cuda.is_available() else -1
    
    start = time.time()
    logger.info(f"Training started for trial {job_number+1}.")

    trainer = Trainer.from_data(
        data_file,
        igpu = igpu,
        max_epoch = max_epoch,
        verbose = verbose,
        work_dir = work_dir,
        config_parameters = trainer_config,
        logger = logger
    )
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(timeout_hours * 3600))

    try:
        metrics = trainer.train()
        logger.info(metrics)
        n_aux = trainer_config.get("n_aux", 0)
        trainer.test_models(data_file, n_aux=n_aux, work_dir=work_dir)
    except Exception as e:
        logger.warn(f"Error happened: {e.args}")
        metrics = e.args
    signal.alarm(0)
    
    time_used = time.time() - start
    logger.info(f"Training finished. Time used: {time_used:.2f}s.\n\n")
    
    return metrics, time_used


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_file', type=str, required=True,
                        help='File name of the dataset in CSV format')
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='Config for training parameter in YAML format')
    parser.add_argument('-e', '--max_epoch', type=int, default=2000,
                        help='Maximum iterations')
    parser.add_argument('-v', '--verbose', action="store_true",
                        help='Maximum iterations')
    parser.add_argument('-w', "--work_dir", type=str, default='.',
                        help="Working directory to write the output files")
    parser.add_argument('--trials', type=int, default=1,
                        help='Total number of trainings to run')
    parser.add_argument('--timeout', type=int, default=5,
                        help='Time limit per job in hours')
    args = parser.parse_args()
    max_epoch = args.max_epoch
    verbose = args.verbose
    trials = args.trials
    work_dir = os.path.abspath(os.path.expanduser(args.work_dir))
    assert os.path.exists(work_dir)
    data_file = os.path.join(work_dir, args.data_file)
    trainer_config = Parameters.from_yaml(os.path.join(work_dir, args.config))

    # Start Logger
    logger = create_logger("Main training:", f'{work_dir}/main_process_message.txt', append=True)
    logger.info("START")

    if trials > 1:
        par_map, nprocesses = get_parallel_map_func(work_dir, logger=logger)
    else:
        par_map, nprocesses = map, 1
    logger.info("Running with {} process(es).".format(nprocesses))
    
    start = time.time()
    result = par_map(
        run_training,
        list(range(trials)),
        [work_dir] * trials,
        [trainer_config] * trials,
        [max_epoch] * trials,
        [verbose] * trials,
        [data_file] * trials,
        [args.timeout] * trials,
        [logger] * trials
    )

    time_trials = np.array([r[1] for r in list(result)])
    logger.info(
        f"Time used for each trial: {time_trials.mean():.2f} +/- {time_trials.std():.2f}s.\n" + 
        ' '.join([f"{t:.2f}s" for t in time_trials])
    )
    
    end = time.time()
    logger.info(
        f"Total time used: {end-start:.2f}s for {trials} trails " +
        f"({(end-start)/trials:.2f} each on average)."
    )
    logger.info("END\n\n")

if __name__ == '__main__':
    main()
