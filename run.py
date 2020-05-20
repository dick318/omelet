import warnings
warnings.simplefilter("ignore")

import os
import sys
import time
import argparse
from omelet.app import run
from tabulate import tabulate


if __name__=='__main__':
    os.system('cls' if os.name == 'nt' else 'clear')

    desc="""\

    ▄▄▄▄▄▄▄ ▄▄   ▄▄ ▄▄▄▄▄▄▄ ▄▄▄     ▄▄▄▄▄▄▄ ▄▄▄▄▄▄▄      ▄▄   ▄▄ ▄▄▄▄▄▄▄
    █       █  █▄█  █       █   █   █       █       █    █  █ █  █       █
    █   ▄   █       █    ▄▄▄█   █   █    ▄▄▄█▄     ▄█    █  █▄█  █▄▄▄▄   █
    █  █ █  █       █   █▄▄▄█   █   █   █▄▄▄  █   █      █   █   █▄▄▄▄█  █
    █  █▄█  █       █    ▄▄▄█   █▄▄▄█    ▄▄▄█ █   █  ▄▄▄ █       █ ▄▄▄▄▄▄█
    █       █ ██▄██ █   █▄▄▄█       █   █▄▄▄  █   █ █   █ █▄   ▄██ █▄▄▄▄▄
    █▄▄▄▄▄▄▄█▄█   █▄█▄▄▄▄▄▄▄█▄▄▄▄▄▄▄█▄▄▄▄▄▄▄█ █▄▄▄█ █▄▄▄█   █▄█  █▄▄▄▄▄▄▄█

            Automated Machine Learning + XAI + Model Management

    """
    print(desc)

    parser = argparse.ArgumentParser("run.py")

    parser.add_argument('-f', '--file',
                dest='data_file',
                metavar='DATA',
                type=str,
                required=True,
                help="data only supports csv type [required]")

    parser.add_argument('-s', '--sample',
                dest='sample',
                metavar='SAMPLE',
                default=None,
                help="sample number")

    parser.add_argument('-c', '--clf',
                    dest='clf',
                    action="store_true",
                    help="turn on if classifcation")

    parser.add_argument('-p', '--profile',
                dest='profile_file',
                metavar='PROFILE',
                type=str,
                default="default",
                help="set user profile")

    parser.add_argument('-d', '--disabled',
                dest='data_profiling',
                action="store_false",
                help="data profiling off")

    args = parser.parse_args()

    print(" " * 53, end="", flush=True)
    for i in range(6):
        time.sleep(0.65)
        print(".", end="", flush=True)
    print(" connected")
    print("\n" * 3)

    print(' User Arguments Lists')
    # Show User Arguments
    header = ["Argument", "Value"]

    table = [
             ["data_file",             args.data_file],
             ["sample",                args.sample],
             ["cassifier",             args.clf],
             ["profile_file",          args.profile_file],
             ["data_profiling",        args.data_profiling],
            ]

    print(tabulate(table, header, tablefmt="fancy_grid", floatfmt=".8f"))
    time.sleep(3)
    print("\n" * 3)

    run(args.data_file, args.sample, args.clf, args.profile_file, args.data_profiling)
