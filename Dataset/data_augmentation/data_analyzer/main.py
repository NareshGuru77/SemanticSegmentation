from data_augmentation.data_analyzer.data_analysis import DataAnalysis
from data_augmentation.data_analyzer.gather_results import generate_results
from data_augmentation.data_analyzer.data_analysis import variant_to_label_def
from data_augmentation.data_analyzer.plot_results import plot
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

parser.add_argument('--load', default=False, type=bool, required=False,
                    help='Whether to load the results from txt.')

parser.add_argument('--load_file_path', default=None, type=str, required=False,
                    help='Path to the pickle text file.')

args = parser.parse_args()


if not args.load:
    analyzer_tr = DataAnalysis(variant='binary',
                               label_path='/home/nareshguru77/Documents/RnD/final_dataset_results/real_augmented/'
                                          'training/binary')
    analyzer_va = DataAnalysis(variant='binary',
                               label_path='/home/nareshguru77/Documents/RnD/final_dataset_results/real_augmented/'
                                          'validation/binary')
    analyzer_te = DataAnalysis(variant='binary',
                               label_path='/home/nareshguru77/Documents/RnD/final_dataset_results/real_augmented/'
                                          'test/binary')

    results = generate_results(analyzer_tr, analyzer_va, analyzer_te)

    with open('./atWork_binary.txt', 'wb') as f:
        pickle.dump(results, f)

    plot(results)

else:
    with open('./atWork_size.txt', 'rb') as f:
        results = pickle.load(f)

    plot(results)
