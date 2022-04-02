#!/usr/bin/env python

import argparse

import torch
from sc.clustering.trainer import Trainer
from sc.utils.parameter import Parameters
from sc.utils.logger import create_logger
import os
import yaml
import time
import socket
import ipyparallel as ipp
import logging
import signal


engine_id = -1


def timeout_handler(signum, frame):
    raise Exception("Training Overtime!")


def get_parallel_map_func(work_dir="."):
    c = ipp.Client(
        connection_info=f"{work_dir}/ipypar/security/ipcontroller-client.json"
    )

    with c[:].sync_imports():
        from sc.clustering.trainer import Trainer
        import os
        import yaml
        import datetime
        import socket
        import torch
        import sys
        import logging
        import signal
    logging.info(f"Engine IDs: {c.ids}")
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
    if torch.get_num_interop_threads() > 2:
        torch.set_num_interop_threads(1)
        torch.set_num_threads(1)
    
    ngpus_per_node = torch.cuda.device_count()
    if "SLURM_LOCALID" in os.environ:
        local_id = int(os.environ.get("SLURM_LOCALID", 0))
    else:
        local_id = 0
    igpu = local_id % ngpus_per_node if torch.cuda.is_available() else -1

    trainer = Trainer.from_data(
        data_file,
        igpu = igpu,
        max_epoch = max_epoch,
        verbose = verbose,
        work_dir = work_dir,
        config_parameters = trainer_config,
        logger = logger
    )

    start = time.time()
    logger.info(f"Training started.")
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(timeout_hours * 3600))
    try:
        metrics = trainer.train()
        logger.info(metrics)
        logger.info(f"Training finished. Took: {(time.time()-start):.2f}s")
        n_aux = trainer_config.get("n_aux", 0)
        trainer.test_models(data_file, n_aux=n_aux, work_dir=work_dir)
    except Exception as ex:
        logger.warn(f"Error happened: {ex.args}")
        metrics = ex.args
    signal.alarm(0)
    return metrics


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

    work_dir = os.path.abspath(os.path.expanduser(args.work_dir))
    assert os.path.exists(work_dir)
        
    data_file = os.path.join(work_dir, args.data_file)
    trainer_config = Parameters.from_yaml(os.path.join(work_dir, args.config))

    max_epoch = args.max_epoch
    verbose = args.verbose
    trails = args.trials

    logger = create_logger("training", f'{work_dir}/main_process_message.txt')
    logger.info("\nSTART\n\n")

    if trails > 1:
        par_map, nprocesses = get_parallel_map_func(work_dir)
    else:
        par_map, nprocesses = map, 1
    
    logger.info("New job running with {} process(es).".format(nprocesses))
    result = par_map(
        run_training,
        list(range(trails)),
        [work_dir] * trails,
        [trainer_config] * trails,
        [max_epoch] * trails,
        [verbose] * trails,
        [data_file] * trails,
        [args.timeout] * trails
    )
    list(result)
    logger.info("\n\nEND")

if __name__ == '__main__':
    main()
