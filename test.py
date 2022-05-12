from mindspore import dataset as ds
from mindspore.mindrecord import FileReader

M = ds.MindDataset('data/transnasbench/index.mindrecord')
N = FileReader('data/transnasbench/macro.mindrecord')

print(M)