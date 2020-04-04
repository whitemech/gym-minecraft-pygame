#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gym_minecraft_pygame.minecraft_env import ActionSpaceType, MinecraftConfiguration
from gym_minecraft_pygame.wrappers.dict_space import MinecraftDictSpace

import argparse


def parse_arguments():
    parser = argparse.ArgumentParser("minecraft")
    parser.add_argument("--differential", action="store_true", help="Differential action space.")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    action_space_type = ActionSpaceType.DIFFERENTIAL if args.differential else ActionSpaceType.NORMAL
    env = MinecraftDictSpace(MinecraftConfiguration(action_space_type=action_space_type))
    env.play()
