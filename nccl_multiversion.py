#!/usr/bin/env python
# python nccl_multiversion.py

import argparse
import os

import ncluster

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='nccl_multiversion')
parser.add_argument('--instance_type', type=str, default="p3dn.24xlarge")
parser.add_argument('--num_tasks', type=int, default=2, help="number of nodes")
parser.add_argument('--spot', action='store_true', help='use spot instances')
parser.add_argument('--skip_setup', action='store_true',
                    help='can use this option on reruns for slightly faster turn-around')
parser.add_argument('--image_name', type=str, default='dlami23-efa')

parser.add_argument('--force_rebuild', type=int, default=0)
parser.add_argument('--do_efa', type=int, default=1)
parser.add_argument('--do_efa_hack', type=int, default=0)
parser.add_argument('--role', type=str, default='launcher')
parser.add_argument('--nproc_per_node', type=int, default=0, help='number of processes to launch, if not specified, set automatically from number of gpus on instance')
parser.add_argument('--num_gpus', type=int, default=0, help='number of processes to launch, if not specified, set automatically from number of gpus on instance')

args = parser.parse_args()


def launcher():
    #    os.environ['NCLUSTER_AWS_FAST_ROOTDISK'] = '1' # request disk with lots of IOPS on AWS
    job = ncluster.make_job(**vars(args))

    MPI_HOME = "$HOME/anaconda3"
    if not args.nproc_per_node:
        args.nproc_per_node = job.tasks[0].num_gpus
        #    MPI_HOME='/usr/local/mpi'  # for DLAMI 22

    #    job.run('export NCCL_MIN_NRINGS=16')
    job.rsync('.')

    if not job.tasks[0].exists('build_ok') or args.force_rebuild:
        job.run(f'export NCCL_VERSION_TAG="{tag}"')
        job.run(f'export GIT_CHECKOUT_CMD="{gitcmd}"')
        def nccl_build(tag, gitcmd):
            job.run(f'source ~/parameterized_nccl_build.sh')

            #        job.run(f"export MPI_HOME={MPI_HOME}")
        job.run("export NCCL_SOCKET_IFNAME=ens5") # TODO(y): remove because p3dn specific

        should_nccl_build = True

        if should_nccl_build:
            #            nccl_build('2.3.7', "git checkout v2.3.7-1")
            #            nccl_build('2.4.7ms0', "git checkout dev/kwen/multi-socket")
            #            nccl_build('2.4.7', "git checkout v2.4.7-1")
            nccl_build('2.4.6', "git checkout v2.4.6-1")

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
        job.tasks[0].write('build_ok', '0')
    else:
        print("~/build_ok found, skipping setup")


    
    # launch MPI
    hosts = [task.ip for task in job.tasks]
    host_str = ','.join(hosts)

    task0 = job.tasks[0]

    # sanity check, simple mpirun that will print hostnames
    task0.run(f'{MPI_HOME}/bin/mpirun --host {host_str} hostname')

    # TODO: dedup paths logic to nccl_multiversion
    #    tag = '2.4.7'
    #    tag = '2.3.7'
    #    tag = '2.4.7ms0'
    tag = '2.4.6'
    folder_root = f"{task0.homedir}/nccl/nccl-{tag}"
    job.run(f"export NCCL_VERSION_TAG={tag}")
    job.run(f"export FOLDER_ROOT={folder_root}")
    job.run(f"export NCCL_HOME={folder_root}/nccl/build")
    job.run(f"export MPI_HOME=$HOME/anaconda3")
    # CUDA_HOME is used for nccl:nccl-tests:aws-ofi, contains bin, lib, include
    job.run("export CUDA_HOME=/usr/local/cuda-10.0")

    np = args.nproc_per_node

    # https://github.com/NVIDIA/nccl-tests/issues/21
    cmd = (f'{MPI_HOME}/bin/mpirun --host {host_str} -np {args.num_gpus} '
           f'-N 8 '
           f'-mca btl ^openib ' # get rid of no infiniband warning '
           f'-mca oob_tcp_if_include ens5 -mca btl_tcp_if_include ens5 ' # force ens5
           f'-mca orte_base_help_aggregate 0 '   # more logging messages
           f'-x LD_LIBRARY_PATH=~/nccl/nccl-{tag}/nccl/build/lib:$LD_LIBRARY_PATH '
           f'-oversubscribe ' # for "There are not enough slots" error
           f'{folder_root}/nccl-tests/build/all_reduce_perf -b 1280M -e 1280M -f 2 ')

    
    ofi_path=f"~/aws-ofi-nccl/install/lib" # the install script copies libraries into tag-independent location
    nccl_path=f"~/nccl/nccl-{tag}/nccl/build/lib"

    cmd_efa = (f'$HOME/anaconda3/bin/mpirun '
               f'-x FI_PROVIDER="efa" '
               f'-x FI_OFI_RXR_RX_COPY_UNEXP=1 -x FI_OFI_RXR_RX_COPY_OOO=1 '
               f'-x FI_EFA_MR_CACHE_ENABLE=1 -x FI_OFI_RXR_INLINE_MR_ENABLE=1 '
               f'-x LD_LIBRARY_PATH=$HOME/nccl/nccl-$NCCL_VERSION_TAG/aws-ofi-nccl/install/lib/:$NCCL_HOME/lib:/usr/local/cuda-10.0/lib64:/opt/amazon/efa/lib64:$LD_LIBRARY_PATH '
               f'-x NCCL_DEBUG=INFO -x NCCL_TREE_THRESHOLD=0 --host localhost -n 2 -N 2 '
               f'--mca btl tcp,self --mca btl_tcp_if_exclude lo,docker0 --bind-to none '
               f'--oversubscribe '
               f'{folder_root}/nccl-tests/build/all_reduce_perf -b 8 -e 1G -f 2 -g 1 -c 1 -n 2')

    
    # cmd_efa = (f'$HOME/anaconda3/bin/mpirun '
    #            f'-x FI_PROVIDER="efa" '
    #            f'-x FI_OFI_RXR_RX_COPY_UNEXP=1 '
    #            f'-x FI_OFI_RXR_RX_COPY_OOO=1 '
    #            f'-x FI_EFA_MR_CACHE_ENABLE=1 '
    #            f'-x FI_OFI_RXR_INLINE_MR_ENABLE=1 '
    #            f'-x LD_LIBRARY_PATH={ofi_path}:{nccl_path}:/usr/local/cuda-10.0/lib64:/opt/amazon/efa/lib64:$LD_LIBRARY_PATH '
    #            f'-x NCCL_DEBUG=INFO -x '
    #            f'NCCL_TREE_THRESHOLD=0 '
    #            f'--host {host_str} '
    #            f'-n 16 '
    #            f'-N 8 '
    #            f'-oversubscribe ' # for "There are not enough slots" error
    #            '--mca btl tcp,self '
    #            f'--mca btl_tcp_if_exclude lo,docker0 '
    #            f'--bind-to none '
    #            f'~/nccl/nccl-{tag}/nccl-tests/build/all_reduce_perf -b 8 -e 1G -f 2 -g 1 -c 1 -n 16')

    # https://github.com/NVIDIA/nccl-tests/issues/21
    if args.do_efa:
        cmd = cmd_efa
    elif args.do_efa_hack:
        cmd = cmd_efa_hack
    assert not (args.do_efa and args.do_efa_hack)
    
    task0.run(cmd)    

    print(task0.output)


def main():
    if args.role == "launcher":
        launcher()
    elif args.role == "worker":
        assert False, 'unknown arg'
    else:
        assert False, "Unknown role " + args.role


if __name__ == '__main__':
    main()
