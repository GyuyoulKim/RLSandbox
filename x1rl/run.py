import sys
import gym
from collections import defaultdict

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # TODO: solve this with regexes
    env_type = env.entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

def arg_parser():
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def common_arg_parser():
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='BreakoutNoFrameskip-v4')
    parser.add_argument('--play', default=False, action='store_true')
    return parser

def main(args):
    arg_parser = common_arg_parser()
    args, _ = arg_parser.parse_known_args(args)

    from x1rl.dqn.learning_dqn import learn
    learn(args.env)

    return

if __name__ == '__main__':
    main(sys.argv)