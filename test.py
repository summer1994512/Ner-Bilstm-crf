import os
from tqdm import tqdm
import numpy as np
dir_path = os.getcwd()

data = os.path.join(dir_path,'data.txt')

f = open(data, encoding='utf8')
data1 = list()
label = list()
content = f.readlines()
w = len(content)
with open('new_data.txt', 'w', encoding='utf8') as f1:
    for line in tqdm(content):
        a,b = line.strip().lstrip('__label__').split('- , ')
        p = a+'\t'+b.replace(' ','')
        label.append(a)
        data1.append(p)
        f1.write(p)
        f1.write('\n')
print(set(label))
shuffle_indices = np.random.permutation(np.arange(w))
print(np.max(shuffle_indices))
data2 = np.array(data1)[shuffle_indices]

with open('cnews.train.txt', 'w', encoding='utf8') as f2:
    with open('cnews.val.txt', 'w', encoding='utf8') as f3:
        with open('cnews.test.txt', 'w', encoding='utf8') as f4:
            for i,j in tqdm(enumerate(data2)):
                if i<w*0.6:
                    f2.write(j+'\n')
                elif i<w*0.9:
                    f3.write(j+'\n')
                else:
                    f4.write(j+'\n')
