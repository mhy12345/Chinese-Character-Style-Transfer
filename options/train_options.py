from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.
    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--learn_rate', type=float, default=5*1e-5)
        parser.add_argument('--display_freq', type=int, default=20)
        parser.add_argument('--save_freq', type=int, default=500)
        parser.add_argument('--sample_size', type=int, default=8)
        parser.add_argument('--dataset', type=str, default='image_2000x150x64x64_train.npy')
        parser.add_argument('--dataset_mode', type=str, default='cross')

        parser.add_argument('--save_model', type=str, default=True)
        self.isTrain = True
        return parser
