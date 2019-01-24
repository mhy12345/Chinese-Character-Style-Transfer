from .base_dist import BaseDist

class L1Dist(BaseDist):
    def __init__(self):
        BaseDist.__init__(self)

    def __call__(self, real, fake):
        real, fake = real.cpu().detach(), fake.cpu().detach()
        bs, H, W = real.shape
        return (real-fake).abs().view(bs, W*H).mean(1)

