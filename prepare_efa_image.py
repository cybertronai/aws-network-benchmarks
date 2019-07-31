#!/usr/bin/env python
"""

Script that builds EFA-enable image (see nccl_parameterized_build.sh)

# usage (Python 3.6)
pip install -r https://raw.githubusercontent.com/cybertronai/aws-network-benchmarks/master/requirements.txt

export AWS_ACCESS_KEY_ID=<access key id>
export AWS_SECRET_ACCESS_KEY=<secret key>
export AWS_DEFAULT_REGION=us-east-1
export NCLUSTER_ZONE=us-east-1b
"""

import argparse
import os
import re
import shlex
import sys

import util

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='prepare-image')
parser.add_argument('--instance_type', type=str, default="c5.18xlarge")
#  parser.add_argument('--instance_type', type=str, default="p3dn.24xlarge")
parser.add_argument('--num_tasks', type=int, default=1, help="number of nodes")
parser.add_argument('--spot', action='store_true', help='use spot instances')
parser.add_argument('--skip_setup', action='store_true',
                    help='can use this option on reruns for slightly faster turn-around')
parser.add_argument('--image_name', type=str, default='amzn2-ami-hvm-2.0.20190612-x86_64-gp2')

# internal flags
parser.add_argument('--internal_role', type=str, default='launcher')
parser.add_argument('--internal_cmd', type=str, default='echo whoami')
parser.add_argument('--internal_config', type=str, default='800358020000007B7D71002E', help='base16 encoded dict of additional config attributes to log')
parser.add_argument('--internal_config_fn', type=str, default='config_dict', help='location of filename with extra info to log')
parser.add_argument('--ofi_patch_location', type=str, default='')

SETUP_COMPLETED_FN = 'ncluster_setup_completed'

args = parser.parse_args()


def launcher():
    from ncluster import aws_util as u
    import ncluster

    config = {}
    config = vars(args)
    config['disk_size'] = 500
    os.environ['NCLUSTER_AWS_FAST_ROOTDISK'] = '1'
    os.environ['WANDB_SILENT'] = '1'
    task0 = ncluster.make_task(**config)
    task0.rsync('.')

    config['internal_id'] = u.get_account_number()
    config['internal_alias'] = u.get_account_name()
    config['region'] = u.get_region()
    config['zone'] = u.get_zone()
    config['launch_user'] = os.environ.get('USER', '')
    config['launcher_cmd'] = ' '.join([shlex.quote(s) for s in sys.argv])

    
    if os.path.exists(args.ofi_patch_location):
        task0.upload(args.ofi_patch_location)

    pickled_config = util.text_pickle(config)
    task0.write(args.internal_config_fn, pickled_config)

    # make things faster by installing into tmpfs

    INSTALL_ROOT='/home/ec2-user'
    #    task0.run('sudo mkdir -p /tmpfs && sudo chown `whoami` /tmpfs && sudo mount -t tmpfs -o size=50G tmpfs /tmpfs')
    #    INSTALL_ROOT='/tmpfs'
    task0.run(f'export INSTALL_ROOT={INSTALL_ROOT}')
    task0.run(f'export WANDB_SILENT=1')
    
    task0.run(f'mkdir -p {INSTALL_ROOT}/packages')
    task0.run(f'pushd {INSTALL_ROOT}/packages')
    task0.run('sudo yum groupinstall "Development Tools" -y')
    task0.run('sudo update-alternatives --set gcc "/usr/bin/gcc48" || echo ignored')
    task0.run('sudo update-alternatives --set g++ "/usr/bin/g++48" || echo ignored')
    
    task0.run('wget https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh')
    task0.run('bash Anaconda3-2019.03-Linux-x86_64.sh -b || echo ignore')
    task0.run('/home/ec2-user/anaconda3/bin/conda init bash && source ~/.bashrc')
    task0.run('conda create -n pytorch_p36 python=3.6 -y || echo ignore')
    task0.run('source activate pytorch_p36')
    task0.run(f'popd')

    task0.run('pip install -r worker_requirements.txt')
    task0.run(f'python {__file__} --internal_role=worker')


def worker():
    """Runs benchmark locally on AWS and logs results."""
    
    util.install_pdb_handler()
    
    # log info propagated from the launcher
    config = util.text_unpickle(open(args.internal_config_fn).read())
    config['worker_conda'] = util.ossystem('echo ${CONDA_PREFIX:-"$(dirname $(which conda))/../"}')
    config['worker_cmd'] = ' '.join([shlex.quote(s) for s in sys.argv])
    name = f"prepare_efa_image"

    print(config)
    
    import wandb
    # let environment variables override project/run name
    project = None if 'WANDB_ENTITY' in os.environ else 'nccl_bench'
    name = None if 'WANDB_DESCRIPTION' in os.environ else util.get_script_name(__file__)
    wandb.init(project=project, name=name)
    
    wandb.config.update(config)
    if config:
        wandb.config.update(config)

    for key in os.environ:
        if re.match(r"^NCCL|CUDA|PATH|^LD|USER|PWD", key):
            wandb.config['env_'+key] = os.getenv(key)

    util.ossystem2(f'bash indu_build.sh')


    with open(SETUP_COMPLETED_FN, 'w') as f:
        f.write('ok')


def main():
    if args.internal_role == 'launcher':
        launcher()
    elif args.internal_role == 'worker':
        worker()
    else:
        assert False, f'unknown role {args.internal_role}'


if __name__ == '__main__':
    main()
