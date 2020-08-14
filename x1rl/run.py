import sys
import gym
from collections import defaultdict

from x1rl import logger

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
    parser.add_argument('--env',            help='environment ID',              type=str,   default='BreakoutNoFrameskip-v4')
    parser.add_argument('--num_steps',      help='how many steps to run',       type=int,   default=1000000)
    parser.add_argument('--render',         help='render during learning',                  default=False, action='store_true')
    parser.add_argument('--play',           help='play game',                               default=False, action='store_true')
    parser.add_argument('--play_model',     help='play game with this model',   type=str)
    parser.add_argument('--num_episodes',   help='how many episodes to run',    type=int,   default=1)
    return parser

def main(args):
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)

    from x1rl.dqn.learning_dqn import learn, play

    if args.play is True:
        assert args.play_model is not None
        atari_game = args.play_model.split('-')[0] + '-v4'
        logger.record_tabular("play game name:", atari_game)
        logger.dump_tabular()
        play(atari_game, args.play_model, args.num_episodes)
    else:
        learn(args.env, args.num_steps, args.render, unknown_args)

    return

if __name__ == '__main__':
    main(sys.argv)