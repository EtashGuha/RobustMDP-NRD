import argparse
from configargparse import Parser
def get_args():
    parser = Parser(
        args_for_setting_config_path=["-c", "--cfg", "--config"],
        config_arg_is_required=True,
    )
    parser.add_argument('--radius', type=float, default=1.0, help='Name of the model')
    parser.add_argument('--shape', type=int, default=2, help='Name of the model')
    parser.add_argument('--seed', type=int, default=0, help='Name of the model')
    parser.add_argument('--name', type=str, help='Name of the model')



    args = parser.parse_args()
    return args