#!/usr/bin/env python
# python pytorch_bench.py --role=launcher
#
# Most 9.4 Gbps
#
# python pytorch_bench.py --role=launcher

# Older runs:
#
# # 41 Gbps
# # pytorch 1.0, nccl 2.3.7+cuda10.0
# python pytorch_bench.py --nospot --conda_env=pytorch_p36 --role=launcher --name=nt --skip_setup
#
# # 2x layers, 304-308, 50 Gbps
# python pytorch_bench.py --role=launcher --machines=2 --instance_type=p3dn.24xlarge --nproc_per_node=8 --num_rings=16 --num_layers=32

# # 10.7
# # PyTorch 1.1.0a0+3803d1c with nccl 2.3.7
# python pytorch_bench.py --nospot --conda_env=pytorch_april --role=launcher --name=nt --skip_setup
#
# # 12.8
# #  PyTorch 1.1 with nccl 2.4.7ms0
# python pytorch_bench.py --nospot --conda_env=pytorch_april_patched --role=launcher --name=nt --skip_setup

# 16 Rings, 8 Processes, 151-153, 53 Gbps, received 20.9
# 16 rings, 8 processes, 173-178, 46 Gbps, received 20.9
# 171-177ms, 39.8 Gbps
# with nccl 2.4.6 12 Gbps
# python pytorch_bench.py --role=launcher --machines=2 --aws --instance_type=p3dn.24xlarge --nospot --nproc_per_node=8 --num_rings=16 --skip_setup

# 185ms, average bw=28
# python pytorch_bench.py --role=launcher --method=allreduce --machines=2 --aws --instance_type=p3dn.24xlarge --nospot --nproc_per_node=8 --num_rings=16 --skip_setup

# 170ms, average bw=45
# python pytorch_bench.py --role=launcher --machines=2 --aws --instance_type=p3dn.24xlarge --nospot --nproc_per_node=8 --num_rings=16 --skip_setup
#

# with EFA

import wandb
import argparse
import os
import sys
import time
import torch.optim as optim
import torch.distributed as dist

import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

import numpy as np

import torch

# local imports
import util

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='big_pair')
parser.add_argument('--seed', type=int, default=1)

parser.add_argument('--instance_type', type=str, default="p3dn.24xlarge")
parser.add_argument('--num_tasks', type=int, default=2)
parser.add_argument('--nproc_per_node', type=int, default=8)

# pytorch 1.0.1/2.3.7+cuda10.0
parser.add_argument('--conda_env', type=str, default='pytorch_p36')

# pytorch latest/2.3.7+cuda10.0
# parser.add_argument('--conda_env', type=str, default='pytorch_april_nccl237')

# pytorch latest/2.4.6+cuda10.0
# parser.add_argument('--conda_env', type=str, default='pytorch_april')


parser.add_argument('--image_name', type=str, default='Deep Learning AMI (Ubuntu) Version 23.0')

parser.add_argument('--method', type=str, default='optimize')

parser.add_argument('--nospot', action='store_true',
                    help='use regular instead of spot instances')

parser.add_argument('--iters', type=int, default=20,
                    help='how many iterations')
parser.add_argument('--skip_setup', action='store_true')

parser.add_argument('--num_rings', type=int, default=16)
parser.add_argument('--num_layers', type=int, default=16)
parser.add_argument('--bucket_cap', type=int, default=25)

parser.add_argument('--use_latest_nccl', action='store_true')

# worker params
parser.add_argument('--logdir', type=str, default='/tmp')

# distributed params
# TODO: rename worker to launcher
parser.add_argument('--role', type=str, default='worker',
                    help='internal flag, launcher or worker')
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--master_addr', type=str, default='127.0.0.1',
                    help='address of master node')
parser.add_argument('--master_port', type=int, default=-1,
                    help='port of master node')
parser.add_argument('--mpirun', type=int, default=0,
                    help='use mpirun instead of pytorch launcher')
args = parser.parse_args()

fp16 = True
HOSTS_SLOTS_FN = 'hosts.slots'


def _get_nccl_params():
    # from ncluster import aws_util
    params = f'NCCL_DEBUG=VERSION '
    if args.num_tasks > 1:
        params += f'NCCL_MIN_NRINGS={args.num_rings} NCCL_MAX_NRINGS={args.num_rings} '
    #    if aws_util.instance_supports_100gbps_network(args.instance_type):
    #        params += f'NCCL_SOCKET_IFNAME=ens5 '

    return params


def format_env(**d):
    """Converts env var values into variable string, ie
        'var1="val1" var2="val2" '"""
    args_ = [f'{key}="{d[key]}" ' for key in d]
    return ''.join(args_)

def format_env_export(**d):
    """Converts env var values into variable string, ie
        'export var1="val1" && export var2="val2" '"""
    args_ = [f'export {key}="{d[key]}" ' for key in d]
    return ' && '.join(args_)


def format_env_x(**d):
    """Converts env var values into format suitable for mpirun, ie
        '-x var1="val1" -x var2="val2" '"""
    args_ = [f'-x {key}="{d[key]}" ' for key in sorted(d)]
    return ''.join(args_)


def launcher():
    # todo: flag for skip setup

    import ncluster
    job = ncluster.make_job(**vars(args))
    print(f"Logging to {job.logdir}")
    task0 = job.tasks[0]

    nccl_params = _get_nccl_params()

    # pass through launcher params to worker script
    assert '--role=launcher' in sys.argv, "how did you get here?"
    worker_params = sys.argv[1:]
    worker_params.remove('--role=launcher')
    worker_params.extend([f'--logdir {job.logdir}'])

    worker_params = ' '.join(worker_params)  # pass through all args

    dist_params0 = (f'--nproc_per_node={args.nproc_per_node} '
                    f'--nnodes={args.num_tasks} '
                    f'--master_addr={task0.ip} '
                    f'--master_port={6016} ')

    job.rsync('.')
    worker_script_fn = os.path.basename(__file__)  # remote location

    hosts_str, hosts_file_str = util.setup_mpi(job)
    task0.write(HOSTS_SLOTS_FN, hosts_file_str)

    job.run(f'killall -9 python || echo skipping && source activate {args.conda_env}')

    if not args.mpirun:
        for i, task in enumerate(job.tasks):
            dist_params = dist_params0 + f'--node_rank={i} '
            cmd = (f'{nccl_params} python -m torch.distributed.launch {dist_params} {worker_script_fn} '
                   f'{worker_params} ')
            task.run(f'echo {cmd} > {job.logdir}/task-{i}.cmd')  # save command-line
            task.run(cmd, non_blocking=True)
    else:
        # fill in local_rank
        # fill in MASTER_ADDR, MASTER_PORT, WORLD_SIZE
        # OMPI_COMM_WORLD_SIZE
        # OMPI_COMM_WORLD_RANK
        # OMPI_COMM_WORLD_LOCAL_RANK
        # OMPI_COMM_WORLD_NODE_RANK
        FI_PROVIDER = 'efa'
        CUDA_HOME = f'/usr/local/cuda'
        EFA_HOME = f'/opt/amazon/efa'
        MPI_HOME = EFA_HOME
        NUM_GPUS = task0.num_gpus * args.num_tasks
        NPER_NODE = task0.num_gpus

        local_env = format_env_export(LOCAL_RANK='$OMPI_COMM_WORLD_LOCAL_RANK',
                                      RANK='$OMPI_COMM_WORLD_NODE_RANK',
                                      WORLD_SIZE='$OMPI_COMM_WORLD_SIZE',
                                      MASTER_ADDR=task0.ip,
                                      MASTER_PORT=6016)
        mpi_env = format_env_x(FI_PROVIDER=FI_PROVIDER,  # Enables running nccl-tests using EFA provider.
                               FI_OFI_RXR_RX_COPY_UNEXP=1,  #  Disables using bounce buffers for unexpected messages.
                               I_OFI_RXR_RX_COPY_OOO=1,  # Disables using bounce buffers for out of order messages.
                               I_EFA_MR_CACHE_ENABLE=1,  # Enables memory region caching.
                               I_OFI_RXR_INLINE_MR_ENABLE=1,  # Enables inline memory registration of data buffers.
                               CCL_TREE_THRESHOLD=10 * 4294967296,  # force tree for everything under 40GB
                               D_LIBRARY_PATH=f'{CUDA_HOME}/lib:{CUDA_HOME}/lib64:{EFA_HOME}/lib64',
                               NCCL_DEBUG='INFO')

        local_cmd = [f"{local_env} && source activate {args.conda_env} && ",
                     f'python {worker_script_fn} {worker_params} --local_rank="$LOCAL_RANK"']
        local_cmd = ' '.join(local_cmd)

        cmd = [f"{MPI_HOME}/bin/mpirun -n {NUM_GPUS} -N {NPER_NODE} --hostfile {HOSTS_SLOTS_FN} ",
               f'{mpi_env} ',
               f'--mca btl tcp,self --mca btl_tcp_if_exclude lo,docker0 ',
#                f'-mca orte_base_help_aggregate 0 ',   # more logging messages
               f'--bind-to none ',
               f"bash -c '{local_cmd}'"]
        cmd = ' '.join(cmd)

        task0.run(cmd, non_blocking=True)

    task0.join()
    print(task0.output)


class SimpleNet(nn.Module):
    def __init__(self, num_layers, dim):
        super(SimpleNet, self).__init__()
        self.layers = []

        for i in range(num_layers):
            param0 = torch.normal(torch.zeros((dim, dim)), 0.001)
            param = nn.Parameter(param0)
            self.layers.append(param)
            setattr(self, 'W' + str(i), param)

    def forward(self, x):
        for layer in self.layers:
            x = x + layer
        return x


log = None


# noinspection PyArgumentList
def test_optimize():
    global log

    recv_bytes, transmit_bytes = util.network_bytes()

    device = 'cuda'

    dim = 2 ** 12  # multiple of 8, about 67MB matrix in fp32

    model = SimpleNet(args.num_layers, dim)
    model = model.to(device)
    if fp16:
        model = model.half()
        bytes_per_number = 2
    else:
        bytes_per_number = 4

    gradient_size = args.num_layers * (dim * dim) * bytes_per_number
    size_mb = gradient_size / 1e6

    log('initializing process group')
    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            world_size=util.get_world_size())

    log('calling DDP')
    model = DistributedDataParallel(model,
                                    device_ids=[args.local_rank],
                                    output_device=args.local_rank)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    x = torch.eye(dim)
    x = x.to(device)
    if fp16:
        x = x.half()
    time_list = []

    # force initialization of NCCL
    dist.all_reduce(torch.ones(()).cuda())
    dist.barrier()

    log("Start timing")
    start_time = time.perf_counter()
    start_time0 = start_time
    for i in range(args.iters):
        optimizer.zero_grad()

        output = model(x)

        def sqr(a): return a * a

        loss = sqr(output - x).sum()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        elapsed_time_sec = (time.perf_counter() - start_time)
        start_time = time.perf_counter()

        elapsed_time_ms = elapsed_time_sec * 1000
        time_list.append(elapsed_time_ms)
        rate = size_mb / elapsed_time_sec

        log('%03d/%d added %d MBs in %.1f ms: %.2f MB/second %.1f' % (
            i, args.iters, size_mb, elapsed_time_ms, rate, loss))

    del time_list[0]  # first measurement is off because of syncing
    min_time = np.min(time_list)
    median = np.median(time_list)
    log(f"min: {min_time:8.2f}, median: {median:8.2f}, mean: {np.mean(time_list):8.2f}")

    dist.barrier()
    elapsed_time = time.perf_counter() - start_time0
    recv_bytes1, transmit_bytes1 = util.network_bytes()
    log(f"Received {(recv_bytes1 - recv_bytes) / 1e9:.1f}, transmitted {(transmit_bytes1 - transmit_bytes) / 1e9:.1f} "
        f"in {elapsed_time:.1f} seconds")
    log(f"predicted {gradient_size * args.iters / 1e9:.1f}")

    log(f"average observed bw: {(recv_bytes1 - recv_bytes) * 8 / elapsed_time / 1e9:.1f} Gbps")

    time_to_sync_buffer_sec = np.mean(time_list) / 1000
    effective_bw_gbps = gradient_size / time_to_sync_buffer_sec * 8 / 1e9

    log(f"average effective bw: {effective_bw_gbps:0.1f} Gbps")


# noinspection PyArgumentList
def test_allreduce():
    pass
    # global log

    # recv_bytes, transmit_bytes = util.network_bytes()

    # device = 'cuda'

    # dim = 2 ** 12  # multiple of 8, about 67MB matrix in fp32

    # if fp16:
    #     bytes_per_number = 2
    # else:
    #     bytes_per_number = 4

    # gradient_size = args.num_layers * (dim * dim) * bytes_per_number
    # size_mb = gradient_size / 1e6

    # log('initializing process group')
    # dist.init_process_group(backend='nccl',
    #                         init_method='env://',
    #                         world_size=util.get_world_size())

    # xs = [torch.ones((dim, dim)) for _ in range(args.num_layers)]
    # xs = [x.to(device) for x in xs]
    # if fp16:
    #     xs = [x.half() for x in xs]
    # time_list = []

    # # force initialization of NCCL
    # dist.all_reduce(torch.ones(()).cuda())
    # dist.barrier()

    # log("Start timing")
    # start_time = time.perf_counter()
    # start_time0 = start_time
    # for i in range(args.iters):
    #     [dist.all_reduce(x, async_op=True) for x in xs]

    #     torch.cuda.synchronize()
    #     elapsed_time_sec = (time.perf_counter() - start_time)
    #     start_time = time.perf_counter()

    #     elapsed_time_ms = elapsed_time_sec * 1000
    #     time_list.append(elapsed_time_ms)
    #     rate = size_mb / elapsed_time_sec

    #     # could do barrier, but didn't have effect on timing
    #     # dist.barrier()   
    #     new_result = xs[0]
    #     log('%03d/%d added %d MBs in %.1f ms: %.2f MB/second %.1f' % (
    #         i, args.iters, size_mb, elapsed_time_ms, rate, float(new_result[0, 0])))

    # del time_list[0]  # first measurement is off because of syncing
    # min_time = np.min(time_list)
    # median = np.median(time_list)
    # log(f"min: {min_time:8.2f}, median: {median:8.2f}, mean: {np.mean(time_list):8.2f}")

    # dist.barrier()
    # elapsed_time = time.perf_counter() - start_time0
    # recv_bytes1, transmit_bytes1 = util.network_bytes()
    # log(f"Received {(recv_bytes1 - recv_bytes) / 1e9:.1f}, transmitted {(transmit_bytes1 - transmit_bytes) / 1e9:.1f} "
    #     f"in {elapsed_time:.1f} seconds")
    # log(f"predicted {gradient_size * args.iters / 1e9:.1f}")

    # log(f"average observed bw: {(recv_bytes1 - recv_bytes) * 8 / elapsed_time / 1e9:.1f} Gbps")

    # time_to_sync_buffer_sec = np.mean(time_list)/1000
    # effective_bw_gbps = gradient_size/time_to_sync_buffer*8/1e9

    # log(f"average effective bw: {effective_bw_gbps} Gbps")


def main():
    global log
    if args.role == "launcher":
        launcher()
    elif args.role == "worker":
        if os.environ.get('RANK', '0') == '0':
            wandb.init(project='nccl_bench', name='pytorch_bench')

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        log = util.FileLogger(args.logdir + f'/worker-{util.get_global_rank()}', mirror=(args.local_rank == 0))

        torch.cuda.set_device(args.local_rank)
        #      test_p2p()
        if args.method == 'optimize':
            test_optimize()
        elif args.method == 'allreduce':
            test_allreduce()
        else:
            assert False, 'unknown arg'
    else:
        assert False, "Unknown role " + args.role


if __name__ == '__main__':
    main()
