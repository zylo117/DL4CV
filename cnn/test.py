import os
import numpy as np

def get_available_gpus():
    """
    code from http://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
    """
    from tensorflow.python.client import device_lib as _device_lib
    local_device_protos = _device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

# print(get_available_gpus())

env_dist = os.environ # environ是在os.py中定义的一个dict environ = {}

# 打印所有环境变量，遍历字典

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
for key in env_dist:
    print(key + ' : ' + env_dist[key])