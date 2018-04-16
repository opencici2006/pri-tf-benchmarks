#!/usr/bin/python
from argparse import ArgumentParser

class BenchmarkUtil:

  def main(self):
    arg_parser = ArgumentParser(description='Launchpad for Parameterized Docker'
        ' builds')

    arg_parser.add_argument('-bs', "--batch-size",
                            	help="Specify the batch size. If this parameter is not specified " \
				"or is -1, the largest ideal batch size for the model will be used.",
                            	dest="batch_size", type=int, default=-1)
    arg_parser.add_argument('-nc', "--num-cores",
                            help='Specify the number of cores to use. If the parameter is not specified ' \
				'or is -1, all cores will be used.',
                            dest="num_cores", type=int, default=-1)
#This adds support for a --single-socket param with a default value of False. Only if '--single-socket' is on the command-line will the value be true.
    arg_parser.add_argument('-ss','--single-socket', help='Indicates that only one socket should be used. ' \
				'If used in conjunction with --num-cores, all cores will be allocated ' \
				'on the single socket.',
                            dest="single_socket", action='store_true')
#This adds support for a --forward-only param with a default value of False. Only if '--forward-only' is on the command-line will the value be true.
    arg_parser.add_argument('-fo', "--forward-only", help='Only do inference.', dest='forward_only',
				action='store_true')
    arg_parser.add_argument('-cp', "--checkpoint",
                            help='Specify the location of checkpoint/training model. ' \
                                'If --forward-only is not specified, training model/weights will be ' \
                                'written to this location. If --forward-only is specified, ' \
                                'assumes that the location ' \
                                'points to a model that has already been trained. ',
                            dest="checkpoint", default=None)
    arg_parser.add_argument("-dl", "--data-location", help="Specify the location of the data. " \
				"If this parameter is not specified, the benchmark will use random/dummy data.",
                            dest="data_location", default=None)

    self.args = arg_parser.parse_args()

if __name__ == "__main__":
  util = BenchmarkUtil()
  util.main()

