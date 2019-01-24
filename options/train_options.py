from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.
    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--no-load_model', action='store_false', dest='load_model')
        parser.add_argument('--load_model',  action='store_true', dest='load_model')
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--learn_rate', type=float, default=1e-4)
        parser.add_argument('--display_freq', type=int, default=20)
        parser.add_argument('--save_freq', type=int, default=500)
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--sample_size', type=int, default=4)
        parser.add_argument('--dataset', type=str, default='image_2000x150x64x64_train.npy')
        parser.add_argument('--dataset_mode', type=str, default='cross')

        parser.add_argument('--ignore_pattern', type=str, default='')
        parser.add_argument('--no-optm_g', action='store_false', dest='optm_g')
        parser.add_argument('--no-optm_d', action='store_false', dest='optm_d')
        parser.add_argument('--save_model', action='store_true', dest='save_model')
        parser.add_argument('--no-save_model', action='store_false', dest='save_model')
        parser.set_defaults(load_model=True)
        parser.set_defaults(optm_g=True)
        parser.set_defaults(optm_d=True)
        parser.set_defaults(save_model=True)
        self.isTrain = True
        return parser
