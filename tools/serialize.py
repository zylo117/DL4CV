import os, shutil

# input here
path = '../datasets/tripod/'
prefix = 'tripod_'
postfix = ''
index_start = 0


# start here
def num_len(index, length):
    return str(index).zfill(length)


pl = os.listdir(path)

for p in pl:
    try:
        type_path = path + '/' + p
        fl = os.listdir(type_path)
        index = index_start
        for img in fl:
            if img != 'new':
                os.makedirs(type_path + '/new/', exist_ok=True)
                shutil.move(type_path + '/' + img, type_path + '/new/' + prefix + num_len(index, 4) + postfix + '.jpg')
                index += 1

        fl = os.listdir(type_path + '/new/')
        index = index_start
        for img in fl:
            shutil.move(type_path + '/new/' + img, type_path + '/' + prefix + num_len(index, 4) + postfix + '.jpg')
            index += 1
        shutil.rmtree(type_path + '/new/')

    except:
        print('failed to serialize %s' % p)
