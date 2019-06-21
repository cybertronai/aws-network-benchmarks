#!/usr/bin/env python
"""

Script that builds EFA-enable image (see nccl_parameterized_build.sh)

# usage (Python 3.6)
pip install -r https://raw.githubusercontent.com/cybertronai/aws-network-benchmarks/master/requirements.txt

export AWS_ACCESS_KEY_ID=<access key id>
export AWS_SECRET_ACCESS_KEY=<secret key>
export AWS_DEFAULT_REGION=us-east-1
export NCLUSTER_ZONE=us-east-1b
Save AWS's patch to aws-ofi-nccl package into ~/Downloads/aws-ofi-nccl.patch
"""

import argparse
import os
import re
import shlex
import sys

import util

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='prepare-image')
parser.add_argument('--instance_type', type=str, default="c5n.18xlarge")
parser.add_argument('--num_tasks', type=int, default=1, help="number of nodes")
parser.add_argument('--spot', action='store_true', help='use spot instances')
parser.add_argument('--skip_setup', action='store_true',
                    help='can use this option on reruns for slightly faster turn-around')
# parser.add_argument('--image_name', type=str, default="Deep Learning AMI (Amazon Linux) Version 23.0")
parser.add_argument('--image_name', type=str, default='amzn2-ami-hvm-2.0.20190612-x86_64-gp2')
parser.add_argument('--ofi_patch_location', type=str, default=os.environ['HOME']+'/Downloads/aws-ofi-nccl.patch', help='location of patch to apply to aws-ofi install')

# internal flags
parser.add_argument('--internal_role', type=str, default='launcher')
parser.add_argument('--internal_cmd', type=str, default='echo whoami')
parser.add_argument('--internal_config', type=str, default='800358020000007B7D71002E', help='base16 encoded dict of additional config attributes to log')
parser.add_argument('--internal_config_fn', type=str, default='config_dict', help='location of filename with extra info to log')

SETUP_COMPLETED_FN = 'setup_completed'

args = parser.parse_args()


def launcher():
    from ncluster import aws_util as u
    import ncluster

    config = {}
    task0 = ncluster.make_task(**vars(args))
    task0.rsync('.')

    config['internal_id'] = u.get_account_number()
    config['internal_alias'] = u.get_account_name()
    config['region'] = u.get_region()
    config['zone'] = u.get_zone()
    config['launch_user'] = os.environ.get('USER', '')
    config['launcher_cmd'] = ' '.join([shlex.quote(s) for s in sys.argv])

    assert os.path.exists(args.ofi_patch_location)
    task0.upload(args.ofi_patch_location)

    pickled_config = util.text_pickle(config)
    task0.write(args.internal_config_fn, pickled_config)

    # task0.run('pip install -r worker_requirements.txt')  # things needed for worker()
    task0.run('pip install -r worker_requirements.txt --ignore-installed') # get around dlami requiring older version of pyyaml

    cmd = f'bash ~/prepare_efa_image.sh'
    task0.run(f'python {__file__} --internal_role=worker --internal_cmd={shlex.quote(cmd)}')



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
    wandb.init(project='nccl_bench', name=name)
    wandb.config.update(config)
    if config:
        wandb.config.update(config)

    for key in os.environ:
        if re.match(r"^NCCL|CUDA|PATH|^LD|USER|PWD", key):
            wandb.config['env_'+key] = os.getenv(key)

    def nccl_build(nccl_version_tag, gitcmd):
        extra_env = {'NCCL_VERSION_TAG': nccl_version_tag,
                     'GIT_CHECKOUT_CMD': gitcmd,
                     'NCCL_WIPE_PREVIOUS_BUILD': 1}
        #        util.ossystem2(f'bash ~/parameterized_nccl_build.sh',
        #                       extra_env=extra_env)
        util.ossystem2(f'bash ~/indu_build.sh')

        #    nccl_build('2.4.6', "git checkout v2.4.6-1")
        #    nccl_build('2.4.7', "git checkout v2.4.7-1")
    nccl_build('2.4.7ms0', "git checkout dev/kwen/multi-socket")


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
