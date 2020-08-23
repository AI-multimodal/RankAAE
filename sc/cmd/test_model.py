import argparse

from sc.clustering.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_file', type=str, required=True,
                        help='File name of the dataset in CSV format')
    parser.add_argument('-w', '--work_dir', type=str, default='.',
                        help='Working directory')
    parser.add_argument('--final', type=str, default='final.pt',
                        help='File name of final model')
    parser.add_argument('--best', type=str, default='best.pt',
                        help='File name of test model')
    args = parser.parse_args()

    trainer = Trainer.from_data(args.data_file)
    trainer.test_models(args.data_file, work_dir=args.work_dir,
                        final_model_name=args.final, best_model_name=args.best)
