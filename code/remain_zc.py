import glob
import os
import random

num_yc = len(glob.glob('data/xuelang/yc/*'))
list_zc = glob.glob('data/xuelang/zc/*')
random.shuffle(list_zc)
for path in list_zc[num_yc:]:
  os.remove(path)
