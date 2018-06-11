import os
import shutil

segment = 80
path = '../dataset/flowers17'

pl = os.listdir(path)

flower_className = ['Daffodil', 'Snowdrop', 'Lily Valley', 'Bluebell',
                    'Crocus', 'Iris', 'Tigerlily', 'Tulip',
                    'Fritillary', 'Sunflower', 'Daisy', 'Colts\'s Foot',
                    'Dandelion', 'Cowslip', 'Buttercup', 'Windflower', 'Pansy']

for p in pl:
    if '.jpg' in p:
        index = int(p.split("_")[-1].strip(".jpg")) - 1
        classname = index // 80
        classname = flower_className[classname]
        os.makedirs(path + '/' + classname, exist_ok=True)
        shutil.move(path + '/' + p, path + '/' + classname + '/' + p)
