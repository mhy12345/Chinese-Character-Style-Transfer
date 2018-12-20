import logging
import random
import numpy as np
logger = logging.getLogger(__name__)

class SimpleDataset:
    def name(self):
        return 'simple-data-loader'

    def __init__(self):
        pass

    def initialize(self, opt):
        logger.info('Initialize simple-data-loader...')
        self.path = opt.dataset
        self.data = np.load(self.path)
        self.content_size = self.data.shape[0]
        self.style_size = self.data.shape[1]
        self.sample_size = opt.sample_size
        logger.info("Content = %d"%self.content_size)
        logger.info("Style = %d"%self.style_size)
        logger.info("Sample = %d"%self.sample_size)
        logger.info('Initialize finish.')

    def __len__(self):
        return self.content_size * self.style_size

    def __getitem__(self, idx):
        idx1 = idx // self.style_size
        idx2 = idx %  self.style_size
        idxs_1 = [random.randint(0,self.style_size-1) for i in range(self.sample_size)]
        idxs_2 = [random.randint(0,self.content_size-1) for i in range(self.sample_size)]
        return (
                self.data[idx1,idxs_1,:,:],
                self.data[idxs_2,idx2,:,:],
                self.data[idx1,idx2,:,:])
