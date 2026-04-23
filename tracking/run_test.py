import subprocess
import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))  # tracking/
project_root = os.path.dirname(project_root)               # 项目根目录
sys.path.append(project_root)
os.environ["PYTHONPATH"] = project_root + ":" + os.environ.get("PYTHONPATH", "")


# 定义不同的数据集
datasets = [ "uav123", "dtb","uavtrack112",'uav123_10fps','uav123_l','uavdt','visdrone','webuav3m']

tracking_dir = os.path.dirname(os.path.abspath(__file__))
test_script = os.path.join(tracking_dir, "test.py")

# 依次执行 test.py

for dataset in datasets:
    cmd = f"python {test_script} t2track t2track_12 --dataset {dataset} --threads 6 --num_gpus 1"
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True)
