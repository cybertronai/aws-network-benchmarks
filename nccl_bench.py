#!/usr/bin/env python
"""
Run NCCL all_reduce_perf test on regular or EFA-enabled instances

# usage (Python 3.6)
pip install -r https://raw.githubusercontent.com/cybertronai/aws-network-benchmarks/master/requirements.txt

export AWS_ACCESS_KEY_ID=<access key id>
export AWS_SECRET_ACCESS_KEY=<secret key>
export AWS_DEFAULT_REGION=<one of us-east-1, us-west-2, eu-west-1>

# to test EFA
export NCLUSTER_ZONE=us-east-1b
python nccl_multiversion.py --instance_type=p3dn.24xlarge --name=nccl-efa --image_name='dlami23-efa'

# to test Ethernet
python nccl_multiversion.py --instance_type=p3.16xlarge --name=nccl-ethernet --image_name='Deep Learning AMI (Ubuntu) Version 22.0'

"""

import argparse
import os
import re
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
parser.add_argument('--image_name', type=str, default='dlami23-efa', help="Image to use for this run. dlami23-efa was image created by taking DLAMI 23 Amazon version, and installing extra packages in https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start.html")

parser.add_argument('--force_rebuild', type=int, default=0, help="ignore previously build artifacts and rebuild from scratch")
parser.add_argument('--do_efa', type=int, default=-1, help="whether to test EFA setup. If left at -1, determined automatically from instance type.")

# default=os.environ['HOME']+'/Downloads/aws-ofi-nccl.patch'
parser.add_argument('--ofi_patch', type=str, default='', help='local location of patch to apply to aws-ofi install')

# internal flags
parser.add_argument('--internal_role', type=str, default='launcher')
parser.add_argument('--internal_cmd', type=str, default='echo whoami')
parser.add_argument('--internal_config', type=str, default='800358020000007B7D71002E', help='base16 encoded dict of additional config attributes to log')
parser.add_argument('--internal_config_fn', type=str, default='config_dict', help='location of filename with extra info to log')


args = parser.parse_args()

SETUP_COMLETED_FN = 'setup_completed'
HOSTS_SLOTS_FN = 'hosts.slots'

def launcher():
    from ncluster import aws_util as u
    import ncluster

    config = {}
    job = ncluster.make_job(**vars(args))
    task0 = job.tasks[0]
    job.propagate_env(['WANDB_API_KEY'])
    job.rsync('.')
    job.run('pip install -r worker_requirements.txt')  # things needed for worker()

    # choose EFA/no-EFA codepath based on instance-type, overridable by do_efa
    assert args.do_efa in [-1, 0, 1]
    if args.do_efa == -1:
        if u.instance_supports_efa(args.instance_type):
            args.do_efa = 1
        else:
            args.do_efa = 0

    if not job.tasks[0].exists(SETUP_COMLETED_FN) or args.force_rebuild:
        # build nccl versions
        def nccl_build(nccl_version_tag, gitcmd):
            job.run(f'export NCCL_VERSION_TAG="{nccl_version_tag}"')
            job.run(f'export GIT_CHECKOUT_CMD="{gitcmd}"')
            job.run(f'source ~/parameterized_nccl_build.sh')

        nccl_build('2.3.7', "git checkout v2.3.7-1")
        nccl_build('2.4.6', "git checkout v2.4.6-1")
        nccl_build('2.4.7', "git checkout v2.4.7-1")
        nccl_build('2.4.7ms0', "git checkout dev/kwen/multi-socket")

        # setup password-less SSH between all pairs of instances
        public_keys = {}
        for task in job.tasks:
            key_fn = '~/.ssh/id_rsa'  # this fn is special, used by default by ssh
            task.run(f"yes | ssh-keygen -t rsa -f {key_fn} -N ''")

            public_keys[task] = task.read(key_fn + '.pub')

        for task1 in job.tasks:
            task1.run('echo "StrictHostKeyChecking no" >> /etc/ssh/ssh_config',
                      sudo=True, non_blocking=True)
            for task2 in job.tasks:
                # task1 ->ssh-> task2
                task2.run(f'echo "{public_keys[task1]}" >> ~/.ssh/authorized_keys',
                          non_blocking=True)
        
        if args.do_efa == 0:
            job.tasks[0].write(SETUP_COMLETED_FN, 'ok')  # end of no-EFA setup
        job.tasks[0].write(SETUP_COMLETED_FN, '0')
    else:
        print(f"{SETUP_COMLETED_FN} found, skipping setup")

    # create arguments for --hosts {host_str} and --hostfile {HOSTS_SLOTS_FN}
    hosts = [task.ip for task in job.tasks]
    host_str = ','.join(hosts)
    hosts_file_lines = [f'{host} slots=8 max-slots=8' for host in hosts]
    task0.write(HOSTS_SLOTS_FN, '\n'.join(hosts_file_lines))

    #    nccl_version_tag = '2.4.7'
    #    nccl_version_tag = '2.3.7'
    #    nccl_version_tag = '2.4.7ms0'

    CUDA_HOME = f'/usr/local/cuda-10.0'
    MPI_HOME = f'{task0.homedir}/anaconda3'
    NUM_GPUS = 16
    NPER_NODE = NUM_GPUS // 2
    SIZE_MB = 256  # 4096

    config['CUDA_HOME'] = CUDA_HOME
    config['MPI_HOME'] = MPI_HOME
    config['NUM_GPUS'] = NUM_GPUS
    config['NPER_NODE'] = NPER_NODE
    config['SIZE_MB'] = SIZE_MB

    config['do_efa'] = args.do_efa
    config['internal_id'] = u.get_account_number()
    config['internal_alias'] = u.get_account_name()
    config['region'] = u.get_region()
    config['zone'] = u.get_zone()
    config['launch_user'] = os.environ.get('USER', '')
    config['cmd'] = ' '.join(sys.argv)
    config['launcher_conda'] = util.ossystem('echo ${CONDA_PREFIX:-"$(dirname $(which conda))/../"}')
    
    if args.ofi_patch:
        config['ofi_patch_hash'] = hash(open(args.ofi_patch).read())
    
    config.update(vars(args))

    if args.do_efa:
        FI_PROVIDER = 'efa'
    else:
        FI_PROVIDER = 'old'    # this is undefined, so mpirun will fall back to default behavior
    config['network'] = FI_PROVIDER


    if args.ofi_patch:
        assert os.path.exists(args.ofi_patch)
        job.upload(args.ofi_patch)
        config['ofi_patch'] = True
    else:
        config['ofi_patch'] = False
        # delete patch file if present from previous run
        job.run(f'rm -f aws-ofi-nccl.patch')

    if not job.tasks[0].exists(SETUP_COMLETED_FN) or args.force_rebuild:
        config['fresh_build'] = True

        # install rdma core and libibverbs
        job.run('wget http://mirror.centos.org/centos/6/os/x86_64/Packages/rdma-6.9_4.1-3.el6.noarch.rpm')
        job.run('sudo yum install -y rdma-6.9_4.1-3.el6.noarch.rpm')

        job.run('wget http://mirror.centos.org/centos/6/os/x86_64/Packages/libibverbs-1.1.8-4.el6.x86_64.rpm')
        job.run('sudo yum install -y ./libibverbs-1.1.8-4.el6.x86_64.rpm')

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

    if not job.tasks[0].exists(SETUP_COMLETED_FN) or args.force_rebuild:
        # install rdma core and libibverbs
        job.run('wget http://mirror.centos.org/centos/6/os/x86_64/Packages/rdma-6.9_4.1-3.el6.noarch.rpm')
        job.run('sudo yum install -y rdma-6.9_4.1-3.el6.noarch.rpm')

        job.run('wget http://mirror.centos.org/centos/6/os/x86_64/Packages/libibverbs-1.1.8-4.el6.x86_64.rpm')
        job.run('sudo yum install -y ./libibverbs-1.1.8-4.el6.x86_64.rpm')
    job.tasks[0].write(SETUP_COMLETED_FN, 'ok')  # end of EFA setup

    print("Running EFA test")
    NCCL_VERSION_TAG = '2.4.6'
    config['NCCL_VERSION_TAG'] = NCCL_VERSION_TAG
    FOLDER_ROOT = f"{task0.homedir}/nccl/nccl-{NCCL_VERSION_TAG}"
    config['FOLDER_ROOT'] = FOLDER_ROOT
    NCCL_HOME = f'{FOLDER_ROOT}/nccl/build'
    config['NCCL_HOME'] = NCCL_HOME
    EFA_HOME = f'/opt/amazon/efa'
    config['EFA_HOME'] = EFA_HOME

    # sanity check, simple mpirun that will print hostnames
    task0.run(f'{MPI_HOME}/bin/mpirun --host {host_str} hostname')

    # Run through EFA on 2 gpus/2 machines
    #        f'--oversubscribe '  # https://github.com/NVIDIA/nccl-tests/issues/21

    cmd = (f'{MPI_HOME}/bin/mpirun '
           f' -n {NUM_GPUS} -N {NPER_NODE} --hostfile {HOSTS_SLOTS_FN} '
           f'-x FI_PROVIDER="{FI_PROVIDER}" '  # Enables running nccl-tests using EFA provider.
           f'-x FI_OFI_RXR_RX_COPY_UNEXP=1 '  #  Disables using bounce buffers for unexpected messages.
           f'-x FI_OFI_RXR_RX_COPY_OOO=1 '  # Disables using bounce buffers for out of order messages.
           f'-x FI_EFA_MR_CACHE_ENABLE=1 '  # Enables memory region caching.
           f'-x FI_OFI_RXR_INLINE_MR_ENABLE=1 '  # Enables inline memory registration of data buffers.
           f'-x NCCL_TREE_THRESHOLD=4294967296 '
           f'-x LD_LIBRARY_PATH='
           f'{FOLDER_ROOT}/aws-ofi-nccl/install/lib/:'
           f'{NCCL_HOME}/lib:'
           f'{CUDA_HOME}/lib64:'
           f'{EFA_HOME}/lib64:'
           f'{MPI_HOME}/lib:$LD_LIBRARY_PATH '
           f'-x NCCL_DEBUG=INFO '  # print NCCL version config
           f'--mca btl tcp,self --mca btl_tcp_if_exclude lo,docker0 '
           f'--bind-to none '
           f'{FOLDER_ROOT}/nccl-tests/build/all_reduce_perf -b 8 -e {SIZE_MB}M -f 2 -g 1 -c 1 -n 100')

    # assume we didn't change directory from ~
    pickled_config = util.text_pickle(config)
    task0.write(args.internal_config_fn, pickled_config)
    task0.run(f'python {__file__} --internal_role=worker --internal_cmd={shlex.quote(cmd)}')

    print(task0.output)


def worker():
    """Runs benchmark locally on AWS and logs results."""
    
    import wandb
    
    # log info propagated from the launcher
    config = util.text_unpickle(open(args.internal_config_fn).read())
    config['worker_conda'] = util.ossystem('echo ${CONDA_PREFIX:-"$(dirname $(which conda))/../"}')

    num_gpus = 8*args.num_tasks
    patch_str = 'patched' if config['ofi_patch'] else 'stock'
    name = f"bench-{num_gpus}-{config['network']}-{patch_str}"
    wandb.init(project='nccl_bench', name=name)

    # record run config parameters
    print(config)
    wandb.config.update({})
    if config:
        wandb.config.update(config)

    for key in os.environ:
        if re.match(r"^NCCL|CUDA|PATH|^LD|USER|PWD", key):
            wandb.config['env_'+key] = os.getenv(key)

    print("Running command:")
    print(args.internal_cmd)

    output_fn = 'output'
    util.ossystem_with_pipe(args.internal_cmd, output_fn)

    # # get individual bandwidth numbers
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
