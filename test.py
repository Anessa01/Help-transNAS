from cProfile import label
from re import S
from mindspore import dataset as ds
from mindspore.mindrecord import FileReader

M = ds.MindDataset('data/transnasbench/micro.mindrecord')
MD = M.create_dict_iterator()
MT = M.create_tuple_iterator()

fulldict = []
for i in MD:
    dict = {}
    s = i['label']
    dict['dajacency_matrix']=[s[-8] + s[-6] + s[-5] + s[-3] + s[-2] + s[-1]]
    dict['']
    print(i['label'])

print(M)