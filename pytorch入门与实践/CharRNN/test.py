#coding:utf8
import numpy as np
datas = np.load('tang.npz')
data = datas['data']
ix2word = data['ix2word'].item()
poem = data[0]
print(poem)
poem_txt = [ix2word[ii] for ii in poem]
print(''.join(poem_txt))