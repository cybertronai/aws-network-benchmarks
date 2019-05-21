# aws-network-benchmarks
Tools to benchmark AWS network performance, focused on workloads encountered in neural network training.

Goal of these benchmarks is to track/identify bottlenecks that prevent efficient of large neural networks, such as data-parallel training of [Megatron](https://github.com/nvIDIA/Megatron-LM/), which is a 300M parameter BERT model.

## Usage
```
aws configure
pip install -r requirements.txt
<run benchmark>
```

Some benchmarks print result on console, for others, you need to SSH into the machine and look at `sudo nload` to see network usage.

## nccl-tests

This builds latest NCCL and nccl-examples and runs allreduce benchmark.

```
python nccl_multiversion.py  --instance_type=p3dn.24xlarge --nproc_per_node=4
# then SSH into machine and run `sudo nload`, hit Right to see load on ens5
```

Current: 25 Gbps with 16 GPUs over 2 nodes, 1GB allreduce

issues:
- https://github.com/NVIDIA/nccl/issues/209
- https://github.com/NVIDIA/nccl-tests/issues/21

## iperf3

```
python iperf_two_machines.py
# then ssh into machine and run `sudo nload`, hit Right to see load on ens5
```

Current: 91-93 Gbps with 8 processes/10 connections each

## PyTorch/nccl

```
python pytorch_bench.py --role=launcher
```

Issues
- https://github.com/NVIDIA/nccl/issues/209


Current:
- using NCCL 2.3.7: 22.7 Gbps
- using NCCL 2.4+: 9.4 Gbps

## Ray
```
python ray_two_machines_bench.py
```
Current: 11.9 Gbps

Issues:
- https://github.com/ray-project/ray/issues/1325
