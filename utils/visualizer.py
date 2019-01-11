import torch

class Visualizer:
    def __init__(self, max_size = 1000):
        self.q = {}
        self.MAX_SIZE = max_size

    def register(self, key):
        self.q[key] = []

    def update(self, key, value):
        if type(value) == torch.Tensor:
            value = float(value.numpy())
        if len(self.q[key]) == self.MAX_SIZE:
            self.q[key] = self.q[key][1:]
        self.q[key].append(value)

    def __getitem__(self, key):
        if not key in self.q.keys() or len(self.q[key]) == 0:
            return 0.0
        return sum(self.q[key])/len(self.q[key])

    def log(self):
        for k in self.q.keys():
            print(k,' : ', self[k], end='\t')
        print("")
