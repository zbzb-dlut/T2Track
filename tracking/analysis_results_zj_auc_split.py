import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]


from lib.test.analysis.plot_results_zj_auc_split import print_results
from lib.test.evaluation import get_dataset, trackerlist

trackers = []

dataset_name = ('dtb')
dataset_name_result = dataset_name

# choosen from 'lasot_ext_lang', 'lasot_lang', 'otb99_lang', 'tnl2k'

trackers.extend(trackerlist(name='uavtrack', parameter_name='uavtrack_s4', dataset_name=dataset_name,
                            run_ids=None, display_name='vegeta'))

dataset = get_dataset(dataset_name)

print_results(trackers, dataset, dataset_name_result, merge_results=True,seq_eval=True, plot_types=('success', 'prec', 'norm_prec'),
              force_evaluation=True)


