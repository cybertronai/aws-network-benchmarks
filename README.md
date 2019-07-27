# aws-network-benchmarks
Tools to benchmark AWS network performance, focused on workloads encountered in neural network training.

Goal of these benchmarks is to track/identify bottlenecks that prevent efficient of large neural networks, such as data-parallel training of [Megatron](https://github.com/nvIDIA/Megatron-LM/), which is a 300M parameter BERT model.

## Running EFA nccl-test

(tested on fresh instance with DLAMI 23)
```
git clone https://github.com/cybertronai/aws-network-benchmarks
cd aws-network-benchmarks
pip install -r requirements.txt
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-east-1

(optional, to save logs+graphs)
export WANDB_API_KEY=<your key from https://app.wandb.ai/settings>
export NCLUSTER_ZONE=<some zone that contains p3dn instances>

Note: you can use "ncluster spot_prices p3dn" to see valid p3dn zones


python nccl_bench.py --name=test00 --num_tasks=2
```
this launches machines named 0.test00 and 1.test00

to connect to 0.test00 and see logs
```
ncluster connect 0.test00
  or
ssh ec2-user@<ip of 0.test00> -t tmux a
```

This test runs on image prepared using `prepare_efa_image.py` script. Machines stay up indefinitely, kill using `ncluster kill test00` or through AWS console

## Running PyTorch EFA test

Same as above, but use following:
```
python pytorch_bench.py  --role=launcher --num_tasks=2 --mpirun=1 --do_efa=1 --image_name=amzn-efa03
```

# Older stuff

## Usage
```
aws configure
pip install -r requirements.txt
<run benchmark>
```

Some benchmarks print result on console, for others, you need to SSH into the machine and look at `sudo nload` to see network usage.

## nccl-tests

This builds latest NCCL and nccl-examples and runs allreduce benchmark.

For EFA test
```
export NCLUSTER_ZONE=us-east-1b
python nccl_multiversion.py --instance_type=p3dn.24xlarge --name=nccl-efa --image_name='dlami23-efa'
```
For Ethernet test
```
python nccl_multiversion.py --instance_type=p3.16xlarge --name=nccl-ethernet --image_name='Deep Learning AMI (Ubuntu) Version 22.0'
```

Current: EFA=1.35 Gbps, Ethernet= with 16 GPUs over 2 nodes pre-patch 

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
Current: 45.5 Gbps

Issues:
- https://github.com/ray-project/ray/issues/1325
