
import argparse
import os
import random
import string
import sys

import wandb

# in test environments disable pdb intercept
os.environ['NCLUSTER_RUNNING_UNDER_CIRCLECI'] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='mpi_test', help="job name")
parser.add_argument('--instance_type', type=str, default="c5.large")
parser.add_argument('--num_tasks', type=int, default=2)
parser.add_argument('--image_name', type=str, default='Deep Learning AMI (Ubuntu) Version 23.0')
parser.add_argument('--spot', action='store_true',
                    help='use spot instead of regular instances')

parser.add_argument('--nproc_per_node', type=int, default=1)
parser.add_argument('--conda_env', type=str, default='pytorch_p36')

parser.add_argument('--skip_setup', action='store_true')


parser.add_argument('--role', type=str, default='launcher',
                    help='internal flag, launcher or worker')
args = parser.parse_args()


def random_id(k=5):
    """Random id to use for AWS identifiers."""
    #  https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits-in-python
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=k))


def launcher():
    import ncluster
    import util
    job = ncluster.make_job(**vars(args))
    job.rsync('.')
    job.run('pip install -r requirements.txt')
    task0 = job.tasks[0]
    
    hosts_str, hosts_file_str = util.setup_mpi(job, max_slots=1)
    task0.write('hosts.slots', hosts_file_str)
    script_fn = os.path.basename(__file__)
    task0.run(f'mpirun -n 2 -N 1 --hostfile hosts.slots python {script_fn} --role=worker --name={args.name}-{random_id()}', stream_output=True)


def main():
    if args.role == "launcher":
        launcher()
    elif args.role == "worker":
        rank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', 0))  # ompi way
        # rank = int(os.environ.get('RANK', '0'))  # pytorch way
        
        if rank != 0:
            os.environ['WANDB_MODE'] = 'dryrun'  # all wandb.log are no-op
            #        wandb.init(project='aws-network-benchmarks', group=args.name, name='mpi_integration_test', entity='circleci')
        wandb.init(project='aws-network-benchmarks', name=args.name, entity='circleci')
        print(f"{os.uname()[1]} {rank} {' '.join(sys.argv)}")


if __name__ == '__main__':
    main()
