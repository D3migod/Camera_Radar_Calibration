# /usr/bin/python
# coding=utf-8
"""functions for reading yaml.gz files and extracting images grabmsec from threre
"""

from __future__ import print_function

# Попробуй установить cyaml - версию библиотеки парсинга написанную на Си, 
# будет работать в 10 раз быстрее.
# Этот код пытается ее использовать если она в наличии
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

import gzip
import os.path
import numpy as np

#change this to where data is stored
data_dir = '../data'


def ungzip(yaml_in_directory):
    # наши ямлы требуют небольшой ректификации перед использованием
    ungzipped = gzip.open(yaml_in_directory, 'rt')
    ungzipped.readline()
    ungzipped = ungzipped.read()
    ungzipped = ungzipped.replace(':', ': ')

    # собственно парсинг
    yml = yaml.load(ungzipped, Loader=Loader)
    return yml

def read_image_grabmsecs(yml_path):
    yml_data = ungzip(yml_path)
    image_frames = [sh['leftImage']
                    for sh in yml_data['shots'] if 'liftImage' in sh.keys()]
    
    data = np.zeros(shape=(len(image_frames), 1), dtype=int)
    i_real = 0
    for i_fr in image_frames:
        data[i_real, 0] = int(i_fr['grabMsec'])
        i_real += 1

    data = data[:i_real, :]
    return data

if __name__ == '__main__':
    grabmsecs = read_image_grabmsecs(os.path.join(data_dir, 't24.306.026.yaml.gz'))
    print(grabmsecs)
