# -*- coding: utf-8 -*-

"""Minecraft environments using a "dict" state space."""

from gym.spaces import Dict, Discrete

from gym_minecraft_pygame.minecraft_env import Minecraft, MinecraftState, item2int


class MinecraftDictSpace(Minecraft):
    """Minecraft environment with a dictionary space."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._x_space = Discrete(self.configuration.columns)
        self._y_space = Discrete(self.configuration.rows)
        self._theta_space = Discrete(self.configuration.nb_theta)
        self._item_space = Discrete(len(item2int))
        self._command_space = Discrete(3)

        # ** {"task-" + str(i): t.next_action_index for i, t in enumerate(self.task_progresses.values())}

    @property
    def observation_space(self):
        return Dict({
            "x": self._x_space,
            "y": self._y_space,
            "theta": self._theta_space,
            "item": self._item_space,
            "command": self._command_space,
        })

    def observe(self, state: MinecraftState):
        """Observe the state."""
        return state.to_dict()
