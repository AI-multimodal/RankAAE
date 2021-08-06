#!/usr/bin/env python

import argparse

import torch
from sc.clustering.trainer import Trainer
import os
import yaml
import datetime
import socket
import ipyparallel as ipp
import sys
import logging
import signal

engine_id = -1

def timeout_handler(signum, frame):
    raise Exception("Training Overtime!")

def get_parallel_map_func(work_dir="."):
    c = ipp.Client(url_file=f"{work_dir}/ipypar/security/ipcontroller-client.json")
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

def run_training(job_number, work_dir, trainer_config, max_epoch, verbose, data_file, timeout_hours=0):
    work_dir = f'{work_dir}/training/job_{job_number+1}'
    if not os.path.exists(work_dir):
        os.makedirs(work_dir, exist_ok=True)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=f'{work_dir}/messages.txt', level=logging.INFO)
    ngpus_per_node = torch.cuda.device_count()
    if "SLURM_LOCALID" in os.environ:
        local_id = int(os.environ.get("SLURM_LOCALID", 0))
    else:
        local_id = 0
    igpu = local_id % ngpus_per_node if torch.cuda.is_available() else -1

    trainer = Trainer.from_data(data_file,
                                igpu=igpu,
                                max_epoch=max_epoch,
                                verbose=verbose,
                                work_dir=work_dir,
                                **trainer_config)
    t1 = datetime.datetime.now()
    logging.info(f"Training started at {t1} on {socket.gethostname()}")
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(timeout_hours * 3600))
    try:
        metrics = trainer.train()
    except Exception as ex:
        logging.warn(f"Error happened: {ex.args}")
    signal.alarm(0)
    t2 = datetime.datetime.now()
    logging.info(f'training finished at {t2}')
    logging.info(f"Total {(t2 - t1).seconds + (t2 - t1).microseconds * 1.0E-6 :.2f}s used in traing")
    logging.info(metrics)
    n_coord_num = trainer_config.get("n_coord_num", 3)
    n_aux = trainer_config.get("n_aux", 0)
    trainer.test_models(data_file, n_coord_num=n_coord_num, n_aux=n_aux, work_dir=work_dir)
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

    

    work_dir = os.path.expandvars(os.path.expanduser(args.work_dir))
    work_dir = os.path.abspath(work_dir)
    with open(os.path.expandvars(os.path.expanduser(args.config))) as f:
        trainer_config = yaml.full_load(f)

    if not os.path.exists(work_dir):
        os.makedirs(work_dir, exist_ok=True)
    
    max_epoch = args.max_epoch
    verbose = args.verbose
    data_file = os.path.abspath(os.path.expandvars(os.path.expanduser(args.data_file)))
    trails = args.trials

    logging.basicConfig(filename=f'{work_dir}/main_process_message.txt', level=logging.INFO)

    if trails > 1:
        par_map, nprocesses = get_parallel_map_func(work_dir)
    else:
        par_map, nprocesses = map, 1
    logging.info("running with {} processes".format(nprocesses))

    result = par_map(run_training,
                     list(range(trails)), 
                     [work_dir]*trails, 
                     [trainer_config]*trails, 
                     [max_epoch]*trails, 
                     [verbose]*trails, 
                     [data_file]*trails,
                     [args.timeout]*trails)
    list(result)

if __name__ == '__main__':
    main()
