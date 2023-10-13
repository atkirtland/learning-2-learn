import argparse
import sys
from datetime import datetime

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
    parser.add_argument('--max_tasks', type=int, help='Maximum number of tasks')
    parser.add_argument('--overtraining', action='store_true', help='Enable overtraining')
    parser.add_argument('--trialsPerTask', type=int, default=None, help='Number of times to run each task')
    parser.add_argument('--debug_timing', action='store_true', help='Enable debug prints for timing')
    parser.add_argument('--debug_clmult', action='store_true', help='Enable debug prints for multiplication time in CL algo.')
    parser.add_argument('--trialsPerTest',type=int, default=25, help='How often to test the network on the suite of tasks')
    parser.add_argument('--save_dir', type=str, default=datetime.now().replace(microsecond=0).isoformat(), help='Directory in ./data/ in which to store files')
    parser.add_argument('--max_to_keep', type=int, default=None, help='max_to_keep parameter for network Saver')
    parser.add_argument('--replace_test_perfs', action='store_true', help='If flag is set, will save all test perfs to the same file and simply replace it on each new task rather than saving to a new file each time')
    parser.add_argument('--stimLength', type=int, default=500, help='Length of time to show stimulus')
    parser.add_argument('--delayLength', type=int, default=1000, help='Length of delay period')
    parser.add_argument('--decisionLength', type=int, default=500, help='Length of time of decision period')
    parser.add_argument('--uselcp', action='store_true', help="Use the Lipschitz Constant Penalty loss")
    parser.add_argument('--lcplmbda', type=float, default=1.0, help="Lambda value for LCP")
    parser.add_argument('--lcpSampling', action='store_true', help="Use sampling instead of exact gradient for LCP")

    parser.add_argument('--hypers', type=str, default=None, choices=HYPERPARAMS.keys(), help='Argument that sets multiple parameters')

    args = parser.parse_args()

    if args.hypers is not None:
        params = HYPERPARAMS[args.hypers]
        for param, value in params.items():
            setattr(args, param, value)

    if args.max_tasks is None:
        if args.runType == runType.Full:
            args.max_tasks = 31 # originally 1001
        elif args.runType in [runType.DSManifPert, runType.SSManifPert, runType.ControlManifPert]:
            args.max_tasks = 31 # originally 101
        else:
            raise Exception("args.runType is not correct")


    args_dict = vars(args)
    train(**args_dict)