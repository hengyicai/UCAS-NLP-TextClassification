from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument(
            '--model_path',
            type=str,
            required=True,
            help='path to pretrained model'
        )
        self.parser.add_argument(
            '--test_dir',
            type=str,
            required=True,
            help='path to test dir(should have some documents under it)'
        )
        self.parser.add_argument(
            '--suffix_accepted',
            type=str,
            default='txt,TXT',
            help='file with suffix_accepted will be read'
        )
        self.isTrain = False
