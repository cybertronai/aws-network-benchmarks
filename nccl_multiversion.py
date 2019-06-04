#!/usr/bin/env python
"""
Run NCCL all_reduce_perf test on regular or EFA-enabled instances

# usage (Python 3.6)
pip install -r https://raw.githubusercontent.com/cybertronai/aws-network-benchmarks/master/requirements.txt

export AWS_ACCESS_KEY_ID=AKIAIBATdf343
export AWS_SECRET_ACCESS_KEY=z7yKEP/RhO3Olk343aiP
export AWS_DEFAULT_REGION=us-east-1

# to test EFA
export NCLUSTER_ZONE=us-east-1b
python nccl_multiversion.py --instance_type=p3dn.24xlarge --name=nccl-efa --image_name='dlami23-efa'

# to test Ethernet
python nccl_multiversion.py --instance_type=p3.16xlarge --name=nccl-ethernet --image_name='Deep Learning AMI (Ubuntu) Version 22.0'

"""

import argparse
import os
import shlex

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='nccl_multiversion')
parser.add_argument('--instance_type', type=str, default="p3dn.24xlarge")
parser.add_argument('--num_tasks', type=int, default=2, help="number of nodes")
parser.add_argument('--spot', action='store_true', help='use spot instances')
parser.add_argument('--skip_setup', action='store_true',
                    help='can use this option on reruns for slightly faster turn-around')
parser.add_argument('--image_name', type=str, default='dlami23-efa')

parser.add_argument('--force_rebuild', type=int, default=0)
parser.add_argument('--do_efa', type=int, default=-1)
parser.add_argument('--role', type=str, default='launcher')
parser.add_argument('--num_gpus', type=int, default=0,
                    help='number of processes to launch, if not specified, set automatically from number of gpus on instance')
# default=os.environ['HOME']+'/Downloads/aws-ofi-nccl.patch'
parser.add_argument('--ofi_patch', type=str, default='', help='location of patch')

# internal flag
parser.add_argument('--internal_role', type=str, default='launcher')
parser.add_argument('--internal_cmd', type=str, default='echo whoami')


args = parser.parse_args()

SETUP_COMLETED_FN = 'setup_completed'


def launcher():
    from ncluster import aws_util as u
    import ncluster

    job = ncluster.make_job(**vars(args))
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

    # launch MPI
    hosts = [task.ip for task in job.tasks]
    host_str = ','.join(hosts)

    task0 = job.tasks[0]

    #    nccl_version_tag = '2.4.7'
    #    nccl_version_tag = '2.3.7'
    #    nccl_version_tag = '2.4.7ms0'

    CUDA_HOME = f'/usr/local/cuda-10.0'
    MPI_HOME = f'{task0.homedir}/anaconda3'
    NUM_GPUS = 16
    NPER_NODE = NUM_GPUS // 2

    if args.do_efa:
        if args.ofi_patch:
            assert os.path.exists(args.ofi_patch)
            job.upload(args.ofi_patch)
        
        if not job.tasks[0].exists(SETUP_COMLETED_FN) or args.force_rebuild:
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
        FOLDER_ROOT = f"{task0.homedir}/nccl/nccl-{NCCL_VERSION_TAG}"
        NCCL_HOME = f'{FOLDER_ROOT}/nccl/build'
        EFA_HOME = f'/opt/amazon/efa'

        # sanity check, simple mpirun that will print hostnames
        task0.run(f'{MPI_HOME}/bin/mpirun --host {host_str} hostname')

        # Run through EFA on 2 gpus/2 machines
        cmd = (f'{MPI_HOME}/bin/mpirun '
               f' -n {NUM_GPUS} -N {NPER_NODE} '
               f'-x FI_PROVIDER="efa" '  # Enables running nccl-tests using EFA provider.
               f'-x FI_OFI_RXR_RX_COPY_UNEXP=1 '  #  Disables using bounce buffers for unexpected messages.
               f'-x FI_OFI_RXR_RX_COPY_OOO=1 '  # Disables using bounce buffers for out of order messages.
               f'-x FI_EFA_MR_CACHE_ENABLE=1 '  # Enables memory region caching.
               f'-x FI_OFI_RXR_INLINE_MR_ENABLE=1 '  # Enables inline memory registration of data buffers.
               f'-x LD_LIBRARY_PATH='
               f'{FOLDER_ROOT}/aws-ofi-nccl/install/lib/:'
               f'{NCCL_HOME}/lib:'
               f'{CUDA_HOME}/lib64:'
               f'{EFA_HOME}/lib64:'
               f'{MPI_HOME}/lib:$LD_LIBRARY_PATH '
               f'-x NCCL_DEBUG=INFO '  # print NCCL version info
               f'-x NCCL_TREE_THRESHOLD=0 '  # Disable tree-algorithm, faster for <8 instances
               f'--host {host_str} '
               f'--mca btl tcp,self --mca btl_tcp_if_exclude lo,docker0 '
               f'--bind-to none '
               f'--oversubscribe '  # https://github.com/NVIDIA/nccl-tests/issues/21
               f'{FOLDER_ROOT}/nccl-tests/build/all_reduce_perf -b 8 -e 256M -f 2 -g 1 -c 1 ')

        # assume we didn't change directory from ~
        task0.run(f'python {__file__} --internal_role=worker --internal_cmd={shlex.quote(cmd)}')

    else:
        print("Running Ethernet test")
        #  MPI_HOME = '/usr/local/mpi'  # for DLAMI 22
        job.run('export NCCL_MIN_NRINGS=16')  # TODO: move into -x
        # NCCL_VERSION_TAG = '2.3.7' this fails with nccl-test trying to open libmpi.so.40
        NCCL_VERSION_TAG = '2.4.6'
        # fails with [ip-172-31-48-152][[16356,1],5][btl_tcp_endpoint.c:626:mca_btl_tcp_endpoint_recv_connect_ack] received unexpected process identifier [[16356,1],6]
        # [ip-172-31-48-152][[16356,1],2][btl_tcp_endpoint.c:626:mca_btl_tcp_endpoint_recv_connect_ack] received unexpected process identifier [[16356,1],5]

        FOLDER_ROOT = f"{task0.homedir}/nccl/nccl-{NCCL_VERSION_TAG}"
        NCCL_HOME = f'{FOLDER_ROOT}/nccl/build'
        SIZE_MB = 256

        #  about -oversubscribe https://github.com/NVIDIA/nccl-tests/issues/21

        task0.run(f'{MPI_HOME}/bin/mpirun --host {host_str} '
                  f'-np {NUM_GPUS} -N {NPER_NODE} '
                  f'-mca btl ^openib '  # get rid of no infiniband warning '
                  f'-mca orte_base_help_aggregate 0 '   # more logging messages
                  # f'-mca oob_tcp_if_include ens5 -mca btl_tcp_if_include ens5 ' # force ens5 (only use on p3dn + Ethernet)
                  f'-x LD_LIBRARY_PATH='
                  f'{NCCL_HOME}/lib:'
                  f'{CUDA_HOME}/lib64:'
                  f'{MPI_HOME}/lib:$LD_LIBRARY_PATH '
                  f'-oversubscribe '  # for "There are not enough slots" error
                  f'{FOLDER_ROOT}/nccl-tests/build/all_reduce_perf -b 8 -e 1M -f 2 -g 1 -c 1 -n {NUM_GPUS} '
                  f'{FOLDER_ROOT}/nccl-tests/build/all_reduce_perf -b {SIZE_MB}M -e {SIZE_MB}M -f 2 ')

    print(task0.output)


def worker():
    import wandb
    os.system(args.internal_cmd)


def main():
    if args.internal_role == 'launcher':
        launcher()
    elif args.internal_role == 'worker':
        worker()
    else:
        assert False, f'unknown role {args.internal_role}'


if __name__ == '__main__':
    main()
