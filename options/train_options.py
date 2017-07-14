from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument(
            '--corpus_root',
            type=str,
            required=False,
            default='./corpus_test',
            help='path to documents(should have subfolders C1-Class1Name, C2-Class2Name,...,Cn-ClassnName)'
        )
        self.parser.add_argument(
            '--encoding',
            type=str,
            required=False,
            default='gb18030',
            help='file encoding of documents'
        )
        self.parser.add_argument(
            '--tfidf_top_k',
            type=int,
            default=300,
            help='features with tfidf value within top_k will be selected'
        )
        self.parser.add_argument(
            '--path_to_save_model',
            type=str,
            required=True,
            help='path to save the model'
        )
        self.isTrain = True
