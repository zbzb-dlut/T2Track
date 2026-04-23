import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

trackers = []
#"uav123", "dtb","uavtrack112",'uavtrack112_l','uav123_10fps','uav123_l','uavdt','visdrone','webuav3m'
for dataset_name in ["uavtrack112_l","uav123_10fps"]:#'webuav3m'

# choosen from 'uav123', 'nfs', 'lasot_extension_subset', 'lasot', 'otb99_lang', 'tnl2k'

    trackers.extend(trackerlist(name='t2track', parameter_name='t2track_12', dataset_name=dataset_name,
                                run_ids=None, display_name='t2track_12'))

    dataset = get_dataset(dataset_name)

    print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'prec', 'norm_prec'),
                  force_evaluation=True)

