from .base_dist import BaseDist
import torch
import numpy as np
import queue
import ctypes
flood_fill_ll = ctypes.CDLL("./criterions/c/flood_fill.so")
flood_fill_ll.flood_fill.restype = ctypes.c_float

'''
res_int = adder.add_int(4,5)
print("result: " + str(res_int))
'''

def flood_fill(_fake, _real):
    _fake, _real = _fake.cpu().numpy(), _real.cpu().numpy()
    fake_ptr = _fake.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    real_ptr = _real.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    res1 = flood_fill_ll.flood_fill(fake_ptr, real_ptr)
    return res1

def flood_fill_(_fake, _real):
    W,H = _fake.shape
    dis = torch.ones(W,H,dtype=torch.int32)*-1
    q = queue.Queue()
    for i in range(W):
        for j in range(H):
            if _real[i,j] < -.7: #Black Pixel
                q.put((i,j))
                dis[i,j] = 0
    while not q.empty():
        sx,sy = q.get()
        for dx,dy in [(0,1),(0,-1),(-1,0),(1,0)]:
            x = sx + dx
            y = sy + dy
            if x<W and y<H and x>=0 and y>=0 and dis[x,y] == -1:
                dis[x,y] = dis[sx,sy] + 1
                q.put((x,y))
    dlist = []
    for i in range(W):
        for j in range(H):
            if _fake[i,j] < -.7: #Black Pixel
                dlist.append(min(8,dis[i,j].numpy()))
    if len(dlist) == 0:
        return 8
    else:
        return sum(dlist)/len(dlist)

class FloodDist(BaseDist):
    def __init__(self):
        BaseDist.__init__(self)

    def __call__(self, real, fake):
        real, fake = real.cpu().detach(), fake.cpu().detach()
        assert(fake.shape == real.shape)
        bs, W, H = fake.shape
        ans = torch.zeros(bs, dtype=torch.float32)
        for k1 in range(bs):
            _fake = fake[k1,:,:]
            _real = real[k1,:,:]
            ans[k1] = flood_fill(_real, _fake) + flood_fill(_fake, _real) 
            s = (_fake.mean()-_real.mean()).abs().mean()*4
            ans[k1]+=s
        return ans
