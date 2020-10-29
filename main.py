import argparse
import os
import re
import shutil
from pprint import pformat

from distm import dist_entry
from pipeline import pipeline_entry
from utils.config import parse_raw_config
from utils.misc import time_str


def main():
    # parse args
    parser = argparse.ArgumentParser(description='OHL & AWS implementation')
    parser.add_argument('--main_py_rel_path', type=str, required=True)
    parser.add_argument('--config_filename', type=str, required=True)
    parser.add_argument('--exp_dirname', type=str, required=True)
    args = parser.parse_args()
    
    # parse config and get root paths
    cfg = parse_raw_config(path=args.config_filename)
    sh_root = os.getcwd()
    job_name = os.path.split(os.getcwd())[-1]
    cfg.pipeline.common_kwargs['job_name'] = job_name
    exp_root = os.path.join(sh_root, args.exp_dirname)
    cfg.pipeline.common_kwargs['exp_root'] = exp_root
    os.chdir(args.main_py_rel_path)
    proj_root = os.getcwd()
    cfg.pipeline.common_kwargs['meta_tb_lg_root'] = os.path.join(proj_root, 'meta_events', job_name, args.exp_dirname)
    os.chdir(sh_root)
    
    # initialize the distributed env
    dist = dist_entry(cfg.dist)
    if dist.is_master():
        # make exp dir
        try:
            if not os.path.exists(exp_root):
                os.mkdir(exp_root)
        except FileExistsError:
            print(f'{time_str()} unknown bug: exp_root({exp_root}) already exists')
        # backup scripts
        back_dir = os.path.join(exp_root, 'scripts_back')
        shutil.copytree(
            src=proj_root,
            dst=back_dir,
            ignore=shutil.ignore_patterns('*exp*', '.*'),
            ignore_dangling_symlinks=True
        )
        shutil.copytree(
            src=sh_root,
            dst=back_dir + sh_root.replace(proj_root, ''),  # do not use os.path.join, as the slash is preserved after the replacing
            ignore=lambda _, names: {n for n in names if not re.match(r'^(.*)\.(yaml|sh)$', n)},
            ignore_dangling_symlinks=True
        )
        print(f'{time_str()}[rk00] => args: {pformat(args)}\n')
        print(f'{time_str()}[rk00] => raw cfg: {pformat(cfg)}\n')
        print(f'{time_str()}[rk00] => All the scripts are backed up to \'{back_dir}\'.\n')
    
    dist.barrier()
    
    # start up
    pp = pipeline_entry(dist=dist, pp_cfg=cfg.pipeline)
    pp.finalize()
    dist.finalize()


if __name__ == '__main__':
    main()
