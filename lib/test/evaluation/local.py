from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.
    settings.davis_dir = ''
    settings.got10k_lmdb_path = ''
    settings.got10k_path = ''
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_extension_subset_path = ''
    settings.lasot_lmdb_path = ''
    settings.lasot_path = ''
    settings.lasotlang_path = ''
    settings.network_path = '/home/b402/Desktop/B402/2026-1-31/UAVTracking2/T2Track-A1_A7/test/networks'    # Where tracking networks are stored.
    settings.prj_dir = '/home/b402/Desktop/B402/2026-1-31/UAVTracking2/T2Track-A1_A7'
    settings.result_plot_path = '/home/b402/Desktop/B402/2026-1-31/UAVTracking2/T2Track-A1_A7/test/result_plots'
    settings.results_path = '/home/b402/Desktop/B402/2026-1-31/UAVTracking2/T2Track-A1_A7/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/b402/Desktop/B402/2026-1-31/UAVTracking2/T2Track-A1_A7'
    settings.segmentation_path = ''
    settings.tc128_path = ''
    settings.tn_packed_results_path = ''
    settings.vot_path = ''
    settings.tnl2k_path = '/media/b402/d7e1a346-f0db-45b2-a401-980d7a97105f/Data/tnl2k/test'
    settings.trackingnet_path = '/media/b402/d7e1a346-f0db-45b2-a401-980d7a97105f/Data/trackingnet'
    # settings.youtubevos_dir = ''
    settings.uav123_10fps_path = '/media/b402/d7e1a346-f0db-45b2-a401-980d7a97105f/Data/TEST/UAV/UAV123_10fps'
    settings.dtb_path = '/media/b402/d7e1a346-f0db-45b2-a401-980d7a97105f/Data/TEST/DTB70'
    settings.uav123_path = '/media/b402/d7e1a346-f0db-45b2-a401-980d7a97105f/Data/TEST/UAV/UAV123'
    settings.uavtrack_path = '/media/b402/d7e1a346-f0db-45b2-a401-980d7a97105f/Data/TEST/V4RFlight112'
    settings.uavdt_path = '/media/b402/d7e1a346-f0db-45b2-a401-980d7a97105f/Data/TEST/UAVDT/UAVDT_sequences'
    settings.visdrone_path = '/media/b402/d7e1a346-f0db-45b2-a401-980d7a97105f/Data/TEST/VisDrone2019-SOT-test/VisDrone2019-SOT-test-dev'
    settings.uav123_long_path = '/media/b402/d7e1a346-f0db-45b2-a401-980d7a97105f/Data/UAV/UAV123'
    settings.webuav3m_path='/media/b402/d7e1a346-f0db-45b2-a401-980d7a97105f/Data/TEST/WebUAV-3M/Test'
    return settings

