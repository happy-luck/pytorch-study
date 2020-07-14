# #coding:utf8
import numpy as np
import torch as t
datas = np.load('tang.npz',allow_pickle=True)
data = datas['data']
data = t.from_numpy(data)
print(data.size(),data.size(1))
ix2word = datas['ix2word'].item()
word2ix = datas['word2ix'].item()
print(len(ix2word))
print(ix2word[9218])
# poem = data[0]
# print(poem)
# poem_txt = [ix2word[ii] for ii in poem]
# print(''.join(poem_txt))
# 
# str_ = 'abc'
# arr_ = np.array([[1, 2], [3, 4]])
# dict_ = {'a' : 1, 'b': 2}
# np.savez('nn.npz', st= str_, ar = arr_, dic= dict_)

# datas = np.load('nn.npz',allow_pickle=True)
# data = datas['st']
# dic = datas['dic'].item()
# print(dic)