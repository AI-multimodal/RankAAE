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
    verbose, 
    data_file, 
    timeout_hours=0,
    logger = logging.getLogger("training")
):

    work_dir = f'{work_dir}/training/job_{job_number+1}'
    if not os.path.exists(work_dir):
        os.makedirs(work_dir, exist_ok=True)

    # Set up a logger to record general training information
    logger = create_logger(f"subtraining_{job_number+1}", os.path.join(work_dir, "messages.txt"))
    
    # Set up a logger to record losses against epochs during training 
    loss_logger = create_logger(f"losses_{job_number+1}", os.path.join(work_dir, "losses.csv"), simple_fmt=True)

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
        verbose = verbose,
        work_dir = work_dir,
        config_parameters = trainer_config,
        logger = logger,
        loss_logger = loss_logger,
    )
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(timeout_hours * 3600))

    try:
        metrics = trainer.train()
        logger.info(metrics)

    except Exception as e:
        logger.warn(f"Error happened: {e.args}")
        metrics = e.args
    signal.alarm(0)
    
    time_used = time.time() - start
    logger.info(f"Training finished. Time used: {time_used:.2f}s.\n\n")
    
    return metrics, time_used


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='Config for training parameter in YAML format')
    parser.add_argument('-w', "--work_dir", type=str, default='.',
                        help="Working directory to write the output files")
    args = parser.parse_args()

    work_dir = os.path.abspath(os.path.expanduser(args.work_dir))
    trainer_config = Parameters.from_yaml(os.path.join(work_dir, args.config))
    assert os.path.exists(work_dir)

    verbose = trainer_config.get("sys_verbose", False)
    trials = trainer_config.get("sys_trials", 1)
    data_file = os.path.join(work_dir, trainer_config.get("sys_data_file", None))
    timeout = trainer_config.get("sys_timeout", 10)

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
        [verbose] * trials,
        [data_file] * trials,
        [timeout] * trials,
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
