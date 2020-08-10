import argparse
from sc.clustering.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_file', type=str, required=True,
                        help='File name of the dataset in CSV format')
    parser.add_argument('-e', '--max_epoch', type=int, default=2000,
                        help='Maximum iterations')
    parser.add_argument('-v', '--verbose', action="store_true",
                        help='Maximum iterations')
    args = parser.parse_args()

    trainer = Trainer.from_data(args.data_file,
                                max_epoch=args.max_epoch,
                                verbose=args.verbose)
    trainer.train()


if __name__ == '__main__':
    main()
