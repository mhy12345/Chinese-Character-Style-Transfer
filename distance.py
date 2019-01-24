import torch
import numpy as np
import queue
import visdom
from criterions.flood_dist import FloodDist
vis = visdom.Visdom(env='haha')

def demo2():
    data = np.load('./dataset/image_100x100x64x64_shuffled.npy')
    data = data[:1,:]
    tott, tots, W, H = data.shape
    data = torch.tensor(data*2-1)
    vis.images(data[0,:].unsqueeze(1), opts={'captain':'real'}, win='0')
    s = 0
    score = torch.zeros(tott,tots)
    style_dist = FloodDist()

    for s2 in range(tots):
        d1 = data[:,s:s+1]
        d2 = data[:,s2:s2+1]
        score[:,s2] = style_dist(d1,d2).mean()
        #score[:,s2] = (d1-d2).abs().mean()
        print(s,s2,score[:,s2])
    rank = torch.sort(score, 1)[1]
    data = torch.gather(data, 1, rank.view(tott,tots,1,1).expand(tott,tots,W,H))
    vis.images(data[0,:].unsqueeze(1), opts={'captain':'real'}, win='2')
            
demo2()


def demo1():
    data = np.load('../dataset/image_100x100x64x64_shuffled.npy')
    data = torch.tensor(data*2-1)
    s = 0
    t = 0
    real = data[s:s+1,t:t+1]
    fake_1 = data[s+1:s+2,t:t+1] #Same style
    fake_2 = data[s:s+1,t+1:t+2] #Same word

    print('eDist between real & fake_1 :', (real-fake_1).abs().mean())
    print('eDist between real & fake_2 :', (real-fake_2).abs().mean())
    print('sDist between real & fake_1 :', style_dist(real, fake_1))
    print('sDist between real & fake_2 :', style_dist(real, fake_2))

#Word with same style end up been more simular to each other

    vis.image(real[0][0], opts={'captain':'real'}, win='real')
    vis.image(fake_1[0][0], opts={'captain':'fake_1'}, win='fake_1')
    vis.image(fake_2[0][0], opts={'captain':'fake_2'}, win='fake_2')
