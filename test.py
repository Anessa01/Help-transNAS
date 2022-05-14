from cProfile import label
from re import S
from mindspore import dataset as ds
from mindspore.mindrecord import FileReader
import numpy as np
import torch

from data.api import TransNASBenchAPI as API
path2nas_bench_file = "data/api_home/transnas-bench_v10141024.pth"
api = API(path2nas_bench_file)
length = len(api)
task_list = api.task_list # list of tasks
print(f"This API contains {length} architectures in total across {len(task_list)} tasks.")
# This API contains 7352 architectures in total across 7 tasks.

# Check all model encoding
search_spaces = api.search_spaces # list of search space names
all_arch_dict = api.all_arch_dict # {search_space : list_of_architecture_names}
for ss in search_spaces:
   print(f"Search space '{ss}' contains {len(all_arch_dict[ss])} architectures.")
print(f"Names of 7 tasks: {task_list}")
# Search space 'macro' contains 3256 architectures.
# Search space 'micro' contains 4096 architectures.
# Names of 7 tasks: ['class_scene', 'class_object', 'room_layout', 'jigsaw', 'segmentsemantic', 'normal', 'autoencoder']

metrics_dict = api.metrics_dict # {task_name : list_of_metrics}
info_names = api.info_names # list of model info names




# check the training information of the example task
task = "class_object"

print(f"Task {task} recorded the following metrics: {metrics_dict[task]}")
print(f"The following model information are also recorded: {info_names}")
# Task class_object recorded the following metrics: ['train_top1', 'train_top5', 'train_loss', 'valid_top1', 'valid_top5', 'valid_loss', 'test_top1', 'test_top5', 'test_loss', 'time_elapsed']
# The following model information are also recorded: ['inference_time', 'encoder_params', 'model_params', 'model_FLOPs', 'encoder_FLOPs']

L = []
xinfo = 'inference_time'
for xtask in api.task_list:
    for i in range(3256, 7352):
        xarch = api.index2arch(i)
        L.append(api.get_model_info(xarch, xtask, xinfo) * 1000)
    torch.save(L, 'data/transnasbench/latency/' + xtask + '.pt')
    print(xtask + ' saved!')



"""
dict = []
for i in range(3256, 7352):
    idict = {}
    # 0=input, 5=output
    iarch = api.index2arch(i)
    archstr = '0' + iarch[-8] + iarch[-6] + iarch[-5] + iarch[-3] + iarch[-2] + iarch[-1]  + '5'
    arch = []
    for j in archstr:
        arch.append(int(j))
    feature = np.eye(6, dtype=np.float32)[arch]
    idict['adjacency_matrix'] = np.array([[0., 1., 1., 0., 1., 0., 0., 0.],
                                 [0., 0., 0., 1., 0., 1., 0., 0.],
                                 [0., 0., 0., 0., 0., 0., 1., 0.],
                                 [0., 0., 0., 0., 0., 0., 1., 0.],
                                 [0., 0., 0., 0., 0., 0., 0., 1.],
                                 [0., 0., 0., 0., 0., 0., 0., 1.],
                                 [0., 0., 0., 0., 0., 0., 0., 1.],
                                 [0., 0., 0., 0., 0., 0., 0., 0.]],dtype=np.float32)
    idict['operation'] = feature
    dict.append(idict)
    print(i)
torch.save(dict, 'data/transnasbench/architechture.pt')
"""