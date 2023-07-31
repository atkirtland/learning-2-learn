import argparse
import sys

sys.path.append(".")
from train import train, runType

def run_type(value):
    if value not in runType.__members__:
        raise argparse.ArgumentTypeError('runType must be one of Full, DSManifPert, SSManifPert, ControlManifPert')
    return runType[value]

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

HYPERPARAMS = {
    'h1': {'learningRateInit': 0.001, 'beta1': 0.9, 'beta2': 0.999},
    'h2': {'learningRateInit': 0.0001, 'beta1': 0.3, 'beta2': 0.999}
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Parameters')

    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--l2_h', type=float, default=0.0005, help='l2_h firing rate / homeostatic regularizer weight')
    parser.add_argument('--l2_wR', type=float, default=0.001, help='l2_wR, recurrent weight regularizer weight')
    parser.add_argument('--l2_wI', type=float, default=0.0001, help='l2_wI, input weight regularizer weight')
    parser.add_argument('--l2_wO', type=float, default=0.1, help='l2_wO, output regulaizer weight')
    parser.add_argument('--svBnd', type=float, default=10.0, help='svBnd')
    parser.add_argument('--runType', type=run_type, default=runType.Full, help='Run type')
    parser.add_argument('--projGrad', type=str2bool, default=False, help='Whether to project gradient')
    parser.add_argument('--originalAdam', type=str2bool, default=False, help='Whether to use the original Adam optimizer rather than the custom one')
    parser.add_argument('--init_lr_full', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 parameter for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='Beta2 parameter for Adam optimizer')
    parser.add_argument('--max_tasks', type=int, default=30, help='Maximum number of tasks')
    parser.add_argument('--overtraining', action='store_true', help='Enable overtraining')
    parser.add_argument('--trialsPerTask', type=int, default=None, help='Number of times to run each task')
    parser.add_argument('--debug_timing', action='store_true', help='Enable debug prints for timing')
    parser.add_argument('--debug_clmult', action='store_true', help='Enable debug prints for multiplication time in CL algo.')

    parser.add_argument('--hypers', type=str, default=None, choices=HYPERPARAMS.keys(), help='Argument that sets multiple parameters')

    args = parser.parse_args()

    if args.hypers is not None:
        params = HYPERPARAMS[args.hypers]
        for param, value in params.items():
            setattr(args, param, value)

    args_dict = vars(args)
    train(**args_dict)