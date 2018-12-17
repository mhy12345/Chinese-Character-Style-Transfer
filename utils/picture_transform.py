import numpy as np
import os

print('read data...')
raw_data = np.load('../raw_data/image_2939x200x64x64_stand.npy')
raw_data = raw_data.astype(np.float32)

print('to raw...')
np.save('../dataset/image_2939x200x64x64_stand.npy',raw_data)

print("to 100x100...")
data = raw_data[:100,:100,:,:]
np.save('../dataset/image_100x100x64x64_stand.npy',data)

'''
print('to 1000x200x80x80...')
data = raw_data[:1000,:200,:,:]
data = np.pad(data,((0,0),(0,0),(8,8),(8,8)),'constant')
np.save('../dataset/image_1000x200x80x80_stand.npy',data)

print('to 2939x200x80x80...')
data = np.pad(raw_data,((0,0),(0,0),(8,8),(8,8)),'constant')
np.save('../dataset/image_2939x200x80x80_stand.npy',data)
'''
