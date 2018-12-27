class DataPool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, *args):
        if self.pool_size == 0:
            return args
        return_images = []
        for arg in zip(args):
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(arg)
                return_images.append(arg)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    _arg = self.images[random_id].clone()
                    self.images[random_id] = arg
                    return_images.append(_arg)
                else:
                    return_images.append(arg)
        return_images = torch.cat(return_images, 0)
        return return_images

