#!/usr/bin/env python
"""
Run NCCL all_reduce_perf test on regular or EFA-enabled instances.
Customized to use image created by prepare_efa_image.py

# usage (Python 3.6)
pip install -r https://raw.githubusercontent.com/cybertronai/aws-network-benchmarks/master/requirements.txt

export AWS_ACCESS_KEY_ID=<access key id>
export AWS_SECRET_ACCESS_KEY=<secret key>
export AWS_DEFAULT_REGION=us-east-1
export NCLUSTER_ZONE=us-east-1b

Then:
python nccl_bench.py
"""

import argparse
import os
import shlex
import sys

import parse_nccltest_output
import util

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='nccl_bench')
parser.add_argument('--instance_type', type=str, default="p3dn.24xlarge")
parser.add_argument('--num_tasks', type=int, default=2, help="number of nodes")
parser.add_argument('--spot', action='store_true', help='use spot instances')
parser.add_argument('--skip_setup', action='store_true',
                    help='can use this option on reruns for slightly faster turn-around')
parser.add_argument('--image_name', type=str, default='basic-efa01', help="Image to use for this run")
parser.add_argument('--size_mb', type=int, default=1024, help="largest size of allreduce to test")
parser.add_argument('--do_efa', type=int, default=-1, help="whether to test EFA setup. If left at -1, determined automatically from instance type.")
parser.add_argument('--wandb', type=str, default='', help='suffix for wandb log')
parser.add_argument('--custom_ring_order', type=int, default=0, help='use custom ring order from https://github.com/NVIDIA/nccl/issues/209#issuecomment-491475792')
parser.add_argument('--aggregation', type=str, default='tree', choices=('tree', 'ring'), help='aggregation type, ring or tree')

# internal flags
parser.add_argument('--internal_role', type=str, default='launcher')
parser.add_argument('--internal_cmd', type=str, default='echo whoami')
parser.add_argument('--internal_config', type=str, default='800358020000007B7D71002E', help='base16 encoded dict of additional config attributes to log')
parser.add_argument('--internal_config_fn', type=str, default='config_dict', help='location of filename with extra info to log')


args = parser.parse_args()

# use script specific marker to determine when to rebuild
SETUP_COMLETED_FN = util.get_script_name(__file__)+'_setup_completed'
HOSTS_SLOTS_FN = 'hosts.slots'


def launcher():
    from ncluster import aws_util as u
    import ncluster

    config = {}
    job = ncluster.make_job(**vars(args))
    task0 = job.tasks[0]
    job.rsync('.')
    job.run('killall all_reduce_perf || echo nevermind')  # kill previous run
    job.run('pip install -r worker_requirements.txt')     # things needed for worker()
    job.tasks[0].write(SETUP_COMLETED_FN, 'ok')

    # choose EFA/no-EFA codepath based on instance-type, overridable by do_efa
    assert args.do_efa in [-1, 0, 1]
    if args.do_efa == -1:
        if u.instance_supports_efa(args.instance_type):
            args.do_efa = 1
        else:
            args.do_efa = 0

    # setup password-less SSH between all pairs of instances
    hosts_str, hosts_file_str = util.setup_mpi(job)
    task0.write(HOSTS_SLOTS_FN, hosts_file_str)
    
    CUDA_HOME = f'/usr/local/cuda'
    EFA_HOME = f'/opt/amazon/efa'
    NCCL_HOME = f'/usr/local/cuda'
    BENCHMARK_BIN = f'$HOME/packages/nccl-tests/build/all_reduce_perf'
    MPI_HOME = EFA_HOME 
    NUM_GPUS = task0.num_gpus*args.num_tasks
    NPER_NODE = task0.num_gpus
    SIZE_MB = args.size_mb

    config['CUDA_HOME'] = CUDA_HOME
    config['NCCL_HOME'] = '/usr/local/cuda'

    config['do_efa'] = args.do_efa
    config['internal_id'] = u.get_account_number()
    config['internal_alias'] = u.get_account_name()
    config['region'] = u.get_region()
    config['zone'] = u.get_zone()
    config['launch_user'] = os.environ.get('USER', '')
    config['cmd'] = ' '.join(sys.argv)
    config['launcher_conda'] = util.ossystem('echo ${CONDA_PREFIX:-"$(dirname $(which conda))/../"}')
    config['launcher_cmd'] = 'python '+' '.join(sys.argv)
    config['num_gpus'] = NUM_GPUS
    config['wandb_suffix'] = args.wandb
    config.update(vars(args))

    if args.do_efa:
        FI_PROVIDER = 'efa'

        # check that ib_uverbs are loaded, and load them if not
        # Also make sure that EFA provider is available
        for task in job.tasks:
            task.run('/usr/sbin/lsmod')
            if 'verbs' not in task.output:
                task.run('sudo /usr/sbin/modprobe ib_uverbs')
            task.run('/usr/sbin/lsmod')
            assert 'verbs' in task.output

            task.run('/opt/amazon/efa/bin/fi_info -p efa')
            assert 'provider: efa' in task.output
    else:
        FI_PROVIDER = 'sockets'    # this is undefined, so mpirun will fall back to default behavior

    config['network'] = FI_PROVIDER
    print("Running network test")
    
    job.run(f'export EFA_HOME={EFA_HOME}')
    job.run(f'export MPI_HOME={MPI_HOME}')
    job.run(f'export NCCL_HOME={NCCL_HOME}')

    # sanity check, simple mpirun that will print hostnames
    task0.run(f'{MPI_HOME}/bin/mpirun --host {hosts_str} hostname')

    threshold = 0
    if args.aggregation == 'tree':
        threshold = 10*4294967296   # 40 GB

    env = {'FI_PROVIDER': FI_PROVIDER,      # Enables running nccl-tests using EFA provider.
           'FI_OFI_RXR_RX_COPY_UNEXP': 1,   #  Disables using bounce buffers for unexpected messages.
           'FI_OFI_RXR_RX_COPY_OOO': 1,     # Disables using bounce buffers for out of order messages.
           'FI_EFA_MR_CACHE_ENABLE': 1,     # Enables memory region caching.
           'FI_OFI_RXR_INLINE_MR_ENABLE': 1,  # Enables inline memory registration of data buffers.
           'NCCL_TREE_THRESHOLD': threshold,  # switch to rings after this threshold
           'LD_LIBRARY_PATH': f'{CUDA_HOME}/lib:{CUDA_HOME}/lib64:{EFA_HOME}/lib64',
           'NCCL_DEBUG': 'VERSION'
           }

    if args.custom_ring_order:
        env['CUDA_VISIBLE_DEVICES'] = '0,1,3,2,7,6,4,5'

    cmd = (f'{MPI_HOME}/bin/mpirun '
           f' -n {NUM_GPUS} -N {NPER_NODE} --hostfile {HOSTS_SLOTS_FN} '
           f'{util.format_env(env)} '
           f'--mca btl tcp,self --mca btl_tcp_if_exclude lo,docker0 '
           f'-mca orte_base_help_aggregate 0 '   # more logging messages
           f'--bind-to none '
           f'{BENCHMARK_BIN} -b 8 -e {SIZE_MB}M -f 2 -g 1 -c 1 -n 100')

    # assume we didn't change directory from ~
    pickled_config = util.text_pickle(config)
    task0.write(args.internal_config_fn, pickled_config)
    task0.run(f'python {__file__} --internal_role=worker --internal_cmd={shlex.quote(cmd)}')

    print(task0.output)


def worker():
    """Runs benchmark locally on AWS and logs results."""
    
    import wandb

    # log config info propagated from the launcher
    config = util.text_unpickle(open(args.internal_config_fn).read())
    config['worker_conda'] = util.ossystem('echo ${CONDA_PREFIX:-"$(dirname $(which conda))/../"}')

    name = util.get_script_name(__file__)
    if config['wandb_suffix']:
        name = name+'-'+config['wandb_suffix']

    wandb.init(project='nccl_bench', name=name)
    wandb.config.update(config)
    util.log_environment()
    
    print("Running command:")
    print(args.internal_cmd)

    output_fn = 'output'
    util.ossystem_with_pipe(args.internal_cmd, output_fn)

    # get individual bandwidth numbers
    alg_bw, bus_bw, avg_bw = parse_nccltest_output.parse(output_fn)

    wandb.log(parse_nccltest_output.make_readable(alg_bw, 'algbw_'))
    wandb.log(parse_nccltest_output.make_readable(bus_bw, 'busbw_'))
    wandb.log({'avg_bw': avg_bw})


def main():
    if args.internal_role == 'launcher':
        launcher()
    elif args.internal_role == 'worker':
        worker()
    else:
        assert False, f'unknown role {args.internal_role}'


if __name__ == '__main__':
    main()
