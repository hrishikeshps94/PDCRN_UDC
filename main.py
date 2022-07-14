import argparse
import os
from train import Train


def main(arg_list):
    if arg_list.mode == 'train':
        train_pipe = Train(arg_list)
        train_pipe.run()
    elif arg_list.mode == 'test':
        pass
    else:
        raise ValueError('Use a valid mode')




if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="Get command-line arguments.")
    main_path = os.path.dirname(os.path.abspath(__file__))
    arg_parser.add_argument('--mode', type=str, choices=['test', 'train'], default='train')
    arg_parser.add_argument('--checkpoint_folder', type=str, default='checkpoint')
    arg_parser.add_argument('--model_type', type=str, default='PDCRN_SYNTH')
    arg_parser.add_argument('--train_path', type=str, default='ds/train')
    arg_parser.add_argument('--test_path', type=str, default='ds/val')
    arg_parser.add_argument('--batch_size', type=int, default=1)
    arg_parser.add_argument('--epochs', type=int, default=1000)
    arg_parser.add_argument('--LR', type=int, default=1e-4)
    arg_parser.add_argument('--num_filters', type=int, default=64)
    arg_parser.add_argument('--dilation_rates', type=tuple, default=(3, 2, 1, 1, 1, 1))
    arg_parser.add_argument('--nPyramidFilters', type=int, default=64)
    arg_parser.add_argument('--log_name', type=str, default='logger')
    arg_parser.add_argument('--in_ch', type=int, default=3)
    arg_list = arg_parser.parse_args()
    main(arg_list)