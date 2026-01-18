"""
    Implement for running experiment.
    1. Process transfered args/paras data (*.json).
    2. Feed the args/paras into the main().
    3. Runing the experiment and return the results to the 'ExperimentServer'.
    Can be used with: 
    python executor.py --argspath=PATH/TO/args.json --mode=normal/tune
"""
from main import main
import argparse
import json
import nni
import os


def callback_executor(train_opts, func_to_be_run):
    """
    To be called by the other functions.
    Args:
        train_opts (argsparser): Experiment arguements.
        func_to_be_run (function): It's usually the 'main()'

    Returns:
        res: Experiment results
    """
    res = func_to_be_run(train_opts)
    return res

def parse_args(args: dict):
    """
    Convert the args from dict to the namesapce.
    Args:
        args (dict): Experiment arguements.

    Returns:
        _type_: namesapce
    """
    for k in args:
        main_opt.__dict__[k] = args[k]
    main_opt.__delattr__('mode')
    return main_opt


FUNC_TO_RUN = main
INIT_IDX = 0
def callback_executor_tune():
    """
    To be called by the main functions.
    """
    print(main_opt.argspath)
    # assert not os.path.exists(main_opt.argspath)
    args_dict = json.load(open(main_opt.argspath, 'r'))
    train_opts = parse_args(args_dict)
    try:
        tuner_params = nni.get_next_parameter()
        print(tuner_params)
        params = nni.utils.merge_parameter(train_opts, tuner_params)
        FUNC_TO_RUN(params)
    except Exception as e:
        print(111, e)


if __name__ == '__main__':
    import logging
    main_parser = argparse.ArgumentParser(description="Executor")
    main_parser.add_argument('--mode', type=str, default='normal', help='Choose for normal running or auto tuning.')
    main_parser.add_argument('--argspath', type=str, help='Path to the argument file *.json.')
    main_opt = main_parser.parse_args()
    if main_opt.mode == 'normal' or main_opt.mode not in ['normal', 'tune']:
        raise ValueError("mode should be 'tune'")
    callback_executor_tune()

    

    # execute(args_parser)

