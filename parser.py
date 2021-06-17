import argparse
import yaml
import yamlargparse


def generate_new_parser():
    """
    Processing parser
    :return: parser
    """
    parser = yamlargparse.ArgumentParser()

    # Graph
    parser.add_argument('--n', default=1, type=int,  # 5
                        help='Number of graph nodes')
    parser.add_argument('--c', default=5, type=int,  # 155
                        help='Number of channels')
    parser.add_argument('--max_depth', default=5, type=int,  # 5
                        help='Depth of TN')
    parser.add_argument('--borders', default=['2,4'], type=list,
                        help='where to reduce resolution')
    parser.add_argument('--ops', default=['conv_3'], type=list,  # 5
                        help='list of operations')

    # Train
    parser.add_argument('--wlr', default=0.1, type=float,
                        help='Learning rate for optimizing architecture')
    parser.add_argument('--alr', default=0.1, type=float,
                        help='Learning rate for optimizing architecture')
    parser.add_argument('-e', '--max_epoch', default=160, type=int,
                        help='Max epoch')
    parser.add_argument('--batch', default=512, type=int, #25
                        help='Batch size')

    # Regularization
    parser.add_argument('--cutout', default=1, type=int,
                        help='Apply cutout')
    parser.add_argument('--cutout_length', default=16, type=int,
                        help='Cutout length')
    parser.add_argument('--dropout', default=0.2, type=float,
                        help='Dropout probability')
    parser.add_argument('--drop_node', default=0., type=float,
                        help='Probability of dropping a node')
    parser.add_argument('--drop_path', default=0., type=float,
                        help='Probability of dropping a path')

    # General
    parser.add_argument('--gpus', default='3', type=str,
                        help='e.g. 1,2,3')
    parser.add_argument('--num', default=1, type=int,
                        help='Numbering for experiments')
    parser.add_argument('--res_path', default='../result')
    parser.add_argument('--data_path', default='../data')

    # For parsing yaml
    parser.add_argument('--cfg', action=yamlargparse.ActionConfigFile)

    return parser


def get_parser_from_yaml(yaml_file):
    _parser = generate_new_parser()
    _arg = _parser.parse_args(['--cfg', yaml_file])
    return _arg


import os
if __name__ == "__main__":
    parser = generate_new_parser()
    arg = parser.parse_args(['--cfg', '../config/cifar10.yaml'])
    print(arg.cutout_length)

