from data_analyzer.data_analysis import DataAnalysis
from data_analyzer.gather_results import generate_results
from data_analyzer.plot_results import plot_with_area
import pickle
import argparse

parser = argparse.ArgumentParser(
    description='Arguments for data_analysis.')

parser.add_argument('--variant', default='full', type=str, required=False,
                    help='The dataset variant.')

parser.add_argument('--train_label_path', default=None, type=str, required=False,
                    help='label path of training set.')

parser.add_argument('--validation_label_path', default=None, type=str, required=False,
                    help='label path of validation set.')

parser.add_argument('--test_label_path', default=None, type=str, required=False,
                    help='label path of test set.')

parser.add_argument('--load_file', default=False, type=bool, required=False,
                    help='Whether to load the results from txt.')

parser.add_argument('--load_file_path', default=None, type=str, required=False,
                    help='Path to the pickle text file.')

args = parser.parse_args()

print(args.load_file)
if not args.load_file:
    print(True)
    analyzer_tr = DataAnalysis(variant=args.variant,
                               label_path=args.train_label_path)
    analyzer_va = DataAnalysis(variant=args.variant,
                               label_path=args.validation_label_path)
    analyzer_te = DataAnalysis(variant=args.variant,
                               label_path=args.test_label_path)

    results = generate_results(analyzer_tr, analyzer_va, analyzer_te,
                               variant=args.variant)

    with open(args.load_file_path, 'wb') as f:
        pickle.dump(results, f)

    plot_with_area(results, args.variant)

else:
    with open(args.load_file_path, 'rb') as f:
        results = pickle.load(f)

    plot_with_area(results, args.variant)
