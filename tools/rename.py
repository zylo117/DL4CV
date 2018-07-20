import os, shutil

# input here
path = '../datasets/tripod/without_fake'
prefix = 'no_tripod_'
postfix = ''


# start here
def strip_format(filename):
    filename = filename.split('.')
    format = filename[-1]
    filename = '.'.join(filename[:-1])
    return filename, format


pl = os.listdir(path)

for p in pl:
    try:
        filename, format = strip_format(p)
        shutil.move(path + '/' + p, path + '/' + prefix + filename + '.' + postfix + format)

    except:
        print('failed to rename %s' % p)
