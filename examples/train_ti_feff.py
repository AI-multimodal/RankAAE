import argparse
from sc.clustering.trainer import Trainer
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_file', type=str, required=True,
                        help='File name of the dataset in CSV format')
    parser.add_argument('-e', '--max_epoch', type=int, default=2000,
                        help='Maximum iterations')
    parser.add_argument('-v', '--verbose', action="store_true",
                        help='Maximum iterations')
    parser.add_argument('-w', "--work_dir", type=str, default='.',
                        help="Working directory to write the output files")
    args = parser.parse_args()

    work_dir = os.path.expandvars(os.path.expanduser(args.work_dir))
    trainer = Trainer.from_data(args.data_file,
                                max_epoch=args.max_epoch,
                                verbose=args.verbose,
                                work_dir=work_dir)
    metrics = trainer.train()
    print(metrics)
    trainer.test_models(args.data_file, work_dir=work_dir)


if __name__ == '__main__':
    main()
