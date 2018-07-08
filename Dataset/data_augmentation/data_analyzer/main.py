from data_augmentation.data_analyzer.data_analysis import DataAnalysis
from data_augmentation.data_analyzer.gather_results import generate_results
from data_augmentation.data_analyzer.data_analysis import variant_to_label_def
from data_augmentation.data_analyzer.plot_results import plot
import pickle


# analyzer_tr_full = DataAnalysis(VARIANT='full',
#                                 LABEL_PATH='/home/nareshguru77/Documents/RnD/final_dataset_results/real_augmented/training/full')
# analyzer_va_full = DataAnalysis(VARIANT='full',
#                                 LABEL_PATH='/home/nareshguru77/Documents/RnD/final_dataset_results/real_augmented/validation/full')
# analyzer_te_full = DataAnalysis(VARIANT='full',
#                                 LABEL_PATH='/home/nareshguru77/Documents/RnD/final_dataset_results/real_augmented/test/full')
#
# results = generate_results(analyzer_tr_full, analyzer_va_full, analyzer_te_full)
#
# print(results)
#
# with open('./atWork_full.txt', 'wb') as f:
#     pickle.dump(results, f)


with open('./atWork_full.txt', 'rb') as f:
    results = pickle.load(f)

#print(results)
plot(results)
