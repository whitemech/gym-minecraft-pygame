# -*- coding: utf-8 -*-

"""
Minecraft environment with OpenAI Gym interface

Luca Iocchi 2017
"""
import itertools
from abc import ABC, abstractmethod
from collections import OrderedDict
from enum import Enum
from typing import Optional, Tuple, Dict, Union, List, Collection

import gym as gym
import pygame
from gym.spaces import Discrete, MultiDiscrete
import numpy as np

black = [0, 0, 0]
white = [255,255,255]
grey = [180,180,180]
dgrey = [120,120,120]
orange = [180,100,20]
green = [0,200,0]
lgreen = [60,250,60]
dgreen = [0,100,0]
blue = [0,0,250]
lblue = [80,200,200]
brown = [140, 100, 40]
dbrown = [100, 80, 0]
gold = [230, 215, 80]


class Resources(Enum):
    WOOD = "wood"
    GRASS = "grass"
    IRON = "iron"


class Tools(Enum):
    TOOLSHED = "toolshed"
    WORKBENCH = "workbench"
    FACTORY = "factory"


Item = Union[Resources, Tools]


class Action(ABC):

    @property
    def item(self):
        raise NotImplementedError


class Get(Action):

    def __init__(self, resource: Resources):
        self.resource = resource

    def __eq__(self, other):
        return type(self) == type(other) and self.resource == other.resource

    @property
    def item(self) -> Resources:
        return self.resource

class Use(Action):

    def __init__(self, tool: Tools):
        self.tool = tool

    def __eq__(self, other):
        return type(self) == type(other) and self.tool == other.tool

    @property
    def item(self) -> Tools:
        return self.tool


class Task:

    def __init__(self, name: str, actions: Collection[Action]):
        self.name = name
        self.actions = tuple(actions)  # type: Tuple[Action]
        assert len(self.actions) > 0

    def __eq__(self, other):
        return isinstance(other, Task) and self.name == other.name and self.actions == other.actions


class TaskProgress:

    def __init__(self, task: Task):
        self.task = task
        self.next_action_index = 0

    @property
    def next_action(self) -> Optional[Action]:
        """Get the next action to do for this task. None if it is completed."""
        return self.task.actions[self.next_action_index] if self.next_action_index < len(self.task.actions) else None

    @property
    def nb_steps(self):
        return len(self.task.actions)

    def is_good_action(self, action: Action):
        return action == self.next_action

    def do_step(self, action: Action):
        if self.is_good_action(action):
            self.next_action_index += 1

    def is_complete(self):
        return self.next_action_index == len(self.task.actions)

    def reset(self):
        self.next_action_index = 0


TASKS = OrderedDict({
    'make_bridge': Task('make_bridge', [Get(Resources.IRON), Get(Resources.WOOD), Use(Tools.FACTORY)]),
    'make_plank': Task('make_plank', [Get(Resources.WOOD), Use(Tools.TOOLSHED)]),
    'make_cloth': Task('make_cloth', [Get(Resources.GRASS), Use(Tools.FACTORY)]),
    'make_stick': Task('make_stick', [Get(Resources.WOOD), Use(Tools.WORKBENCH)]),
    'make_rope': Task('make_rope',  [Get(Resources.GRASS), Use(Tools.TOOLSHED)]),
    'make_shears': Task('make_shears', [Get(Resources.WOOD), Use(Tools.WORKBENCH), Get(Resources.IRON), Use(Tools.WORKBENCH)]),
    'make_bed': Task('make_bed', [Get(Resources.WOOD), Use(Tools.TOOLSHED), Get(Resources.GRASS), Use(Tools.WORKBENCH)]),
    'make_axe': Task('make_axe', [Get(Resources.WOOD), Use(Tools.WORKBENCH), Get(Resources.IRON), Use(Tools.TOOLSHED)]),
    # 'get_gold': [ ['get_iron', 'get_wood', 'use_factory', 'use_bridge'] ],
    # 'get_gem': [ ['get_wood', 'use_workbench', 'get_iron', 'use_toolshed', 'use_axe'] ]
})

LOCATIONS = [
    (Resources.WOOD, 1, 1),
    (Resources.GRASS, 4, 3),
    (Resources.IRON, 5, 5),
    (Tools.TOOLSHED, 2, 4),
    (Tools.WORKBENCH, 6, 3),
    (Tools.FACTORY, 4, 7)
]

item2location = {item: (x, y) for item, x, y in LOCATIONS}

# add a 'None' item
item2int = dict(map(reversed, enumerate([None] + list(Resources) + list(Tools))))


class Direction:

    NB_DIRECTIONS = 4

    def __init__(self, th: int = 90):
        self.th = th

    def rotate_left(self) -> 'Direction':
        th = (self.th + 90) % 360
        return Direction(th)

    def rotate_right(self) -> 'Direction':
        th = (self.th - 90) % 360
        if th == -90: th = 270
        return Direction(th)


class State(ABC):
    pass


class PygameDrawable(ABC):

    @abstractmethod
    def draw_on_screen(self, screen: pygame.Surface):
        """Draw a Pygame object on a given Pygame screen."""


class _AbstractPygameViewer(ABC):

    @abstractmethod
    def reset(self, state: 'State'):
        pass

    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    def close(self):
        pass


class PygameViewer(_AbstractPygameViewer):

    def __init__(self, state: 'MinecraftState'):
        self.state = state

        pygame.init()
        pygame.display.set_caption('Minecraft')
        self.screen = pygame.display.set_mode([self.state.config.win_width, self.state.config.win_height])
        self.myfont = pygame.font.SysFont("Arial", 30)
        self.drawables = self._init_drawables()  # type: List[PygameDrawable]

    def reset(self, state: 'State'):
        self.state = state
        self.drawables = self._init_drawables()

    def _init_drawables(self) -> List[PygameDrawable]:
        result = list()
        result.append(self.state.grid)
        result.append(self.state.robot)
        return result

    def render(self, mode="human"):
        self._fill_screen()
        self._draw_score_label()
        self._draw_last_command()
        self._draw_game_objects()
        self._draw_task_state()

        if mode == "human":
            pygame.display.update()
        elif mode == "rgb_array":
            screen = pygame.surfarray.array3d(self.screen)
            # swap width with height
            return screen.swapaxes(0, 1)

    def _fill_screen(self):
        self.screen.fill(white)

    def _draw_score_label(self):
        score_label = self.myfont.render(str(self.state.score), 100, pygame.color.THECOLORS['black'])
        self.screen.blit(score_label, (20, 10))

    def _draw_last_command(self):
        cmd = self.state.last_command
        s = '%s' % cmd if cmd else ""
        count_label = self.myfont.render(s, 100, pygame.color.THECOLORS['brown'])
        self.screen.blit(count_label, (60, 10))

    def _draw_game_objects(self):
        for d in self.drawables:
            d.draw_on_screen(self.screen)

    def close(self):
        pygame.display.quit()
        pygame.quit()

    def _draw_task_state(self):
        sinv = ''
        for t in self.state.task_progresses.values():
            if t.is_complete():
                sinv += '*'
            else:
                sinv += '-'

        inv_label = self.myfont.render(sinv, 100, pygame.color.THECOLORS['blue'])
        self.screen.blit(inv_label, (200, 10))


class NormalCommand(Enum):
    NOP = 0
    LEFT = 1
    UP = 2
    RIGHT = 3
    DOWN = 4
    GET = 5
    USE = 6

    def __str__(self):
        cmd = NormalCommand(self.value)
        if cmd == NormalCommand.NOP:
            return "_"
        elif cmd == NormalCommand.LEFT:
            return "<"
        elif cmd == NormalCommand.RIGHT:
            return ">"
        elif cmd == NormalCommand.UP:
            return "^"
        elif cmd == NormalCommand.DOWN:
            return "v"
        elif cmd == NormalCommand.GET:
            return "g"
        elif cmd == NormalCommand.USE:
            return "u"
        else:
            raise ValueError("Shouldn't be here...")


class DifferentialCommand(Enum):
    NOP = 0
    LEFT = 1
    FORWARD = 2
    RIGHT = 3
    BACKWARD = 4
    GET = 5
    USE = 6

    def __str__(self):
        cmd = DifferentialCommand(self.value)
        if cmd == DifferentialCommand.NOP:
            return "_"
        elif cmd == DifferentialCommand.LEFT:
            return "<"
        elif cmd == DifferentialCommand.RIGHT:
            return ">"
        elif cmd == DifferentialCommand.FORWARD:
            return "^"
        elif cmd == DifferentialCommand.BACKWARD:
            return "v"
        elif cmd == DifferentialCommand.GET:
            return "g"
        elif cmd == DifferentialCommand.USE:
            return "u"
        else:
            raise ValueError("Shouldn't be here...")


class TeleportCommand(Enum):
    NOP = 0
    GOTO_WOOD = 1
    GOTO_GRASS = 2
    GOTO_IRON = 3
    GOTO_TOOLSHED = 4
    GOTO_WORKBENCH = 5
    GOTO_FACTORY = 6
    GET = 7
    USE = 8

    def __str__(self):
        cmd = TeleportCommand(self.value)
        if cmd == TeleportCommand.NOP:
            return "_"
        elif cmd == cmd.GOTO_WOOD:
            return "goto_wood"
        elif cmd == cmd.GOTO_GRASS:
            return "goto_grass"
        elif cmd == cmd.GOTO_IRON:
            return "goto_iron"
        elif cmd == cmd.GOTO_TOOLSHED:
            return "goto_toolshed"
        elif cmd == cmd.GOTO_WORKBENCH:
            return "goto_workbench"
        elif cmd == cmd.GOTO_FACTORY:
            return "goto_factory"
        elif cmd == cmd.GET:
            return "g"
        elif cmd == cmd.USE:
            return "u"
        else:
            raise ValueError("Shouldn't be here...")


class ActionSpaceType(Enum):
    NORMAL = "normal"
    DIFFERENTIAL = "differential"
    TELEPORT = "teleport"


class MinecraftConfiguration:

    def __init__(self, horizon: Optional[int] = None,
                 nb_goals: int = len(TASKS),
                 action_space_type = ActionSpaceType.NORMAL,
                 reward_outside_grid: float = -1.0,
                 reward_bad_get: float = -1.0,
                 reward_bad_use: float = -1.0,
                 reward_per_step: float = -0.01):
        assert 1 <= nb_goals <= len(TASKS), "at least 1 goal and at most 8."
        self.nb_goals = nb_goals
        self.action_space_type = ActionSpaceType(action_space_type)
        self._tasks = tuple(TASKS.values())[:self.nb_goals]
        self._horizon = horizon if horizon else (self.columns * self.rows) * 10
        self.reward_outside_grid = reward_outside_grid
        self.reward_bad_get = reward_bad_get
        self.reward_bad_use = reward_bad_use
        self.reward_per_step = reward_per_step

        self.offx = 40
        self.offy = 100
        self.radius = 5
        self.size_square = 40

    @property
    def rows(self):
        return 10

    @property
    def columns(self):
        return 10

    @property
    def win_width(self):
        if self.columns > 10:
            return self.size_square * (self.columns - 10)
        else:
            return 480

    @property
    def win_height(self):
        if self.rows > 10:
            return self.size_square * (self.rows-10)
        else:
            return 520

    @property
    def action_space(self):
        if self.action_space_type == ActionSpaceType.NORMAL:
            return Discrete(len(NormalCommand))
        elif self.action_space_type == ActionSpaceType.DIFFERENTIAL:
            return Discrete(len(DifferentialCommand))
        elif self.action_space_type == ActionSpaceType.TELEPORT:
            return Discrete(len(TeleportCommand))
        else:
            raise ValueError("Action space type not recognized.")

    @property
    def observation_space(self):
        if self.action_space_type == ActionSpaceType.DIFFERENTIAL:
            # 4 is the number of possible direction - nord, sud, west, east
            return MultiDiscrete((self.columns, self.rows, Direction.NB_DIRECTIONS))
        else:
            return MultiDiscrete((self.columns, self.rows))

    def get_action(self, action: int) -> Union[DifferentialCommand, NormalCommand, TeleportCommand]:
        if self.action_space_type == ActionSpaceType.NORMAL:
            return NormalCommand(action)
        elif self.action_space_type == ActionSpaceType.DIFFERENTIAL:
            return DifferentialCommand(action)
        elif self.action_space_type == ActionSpaceType.TELEPORT:
            return TeleportCommand(action)
        else:
            raise ValueError("Action space type not recognized.")

    @property
    def horizon(self):
        return self._horizon

    @property
    def nb_theta(self):
        return Direction.NB_DIRECTIONS

    @property
    def tasks(self):
        return self._tasks


class Robot(PygameDrawable):

    def __init__(self, config: MinecraftConfiguration):
        self.config = config

        self._initial_x = 3
        self._initial_y = 2
        self._initial_th = 90

        self.x = self._initial_x
        self.y = self._initial_y
        self.direction = Direction(self._initial_th)

    @property
    def position(self) -> Tuple[int, int]:
        return self.x, self.y

    def reset(self):
        self.x = self._initial_x
        self.y = self._initial_y
        self.direction = Direction(self._initial_th)

    def draw_on_screen(self, screen: pygame.Surface):
        dx = int(self.config.offx + self.x * self.config.size_square)
        dy = int(self.config.offy + (self.config.rows - self.y - 1) * self.config.size_square)
        pygame.draw.circle(screen, pygame.color.THECOLORS['orange'],
                           [dx + self.config.size_square // 2, dy + self.config.size_square // 2],
                           2 * self.config.radius, 0)
        ox = 0
        oy = 0
        if self.direction.th == 0:  # right
            ox = self.config.radius
        elif self.direction.th == 90:  # up
            oy = -self.config.radius
        elif self.direction.th == 180:  # left
            ox = -self.config.radius
        elif self.direction.th == 270:  # down
            oy = self.config.radius

        pygame.draw.circle(screen, pygame.color.THECOLORS['black'],
                           [dx + self.config.size_square // 2 + ox, dy + self.config.size_square // 2 + oy], 5, 0)

    def step(self, command: Union[DifferentialCommand, NormalCommand, TeleportCommand]):
        if isinstance(command, NormalCommand):
            self._step_normal(command)
        elif isinstance(command, DifferentialCommand):
            self._step_differential(command)
        elif isinstance(command, TeleportCommand):
            self._step_teleport(command)
        else:
            raise ValueError("Command not recognized.")

    def _step_normal(self, command: NormalCommand):
        if command == command.DOWN:
            self.y -= 1
        elif command == command.UP:
            self.y += 1
        elif command == command.RIGHT:
            self.x += 1
        elif command == command.LEFT:
            self.x -= 1

    def _step_differential(self, command: DifferentialCommand):
        dx = 1 if self.direction.th == 0 else -1 if self.direction.th == 180 else 0
        dy = 1 if self.direction.th == 90 else -1 if self.direction.th == 270 else 0
        if command == command.LEFT:
            self.direction = self.direction.rotate_left()
        elif command == command.RIGHT:
            self.direction = self.direction.rotate_right()
        elif command == command.FORWARD:
            self.x += dx
            self.y += dy
        elif command == command.BACKWARD:
            self.x -= dx
            self.y -= dy

    def _step_teleport(self, command: TeleportCommand):
        if command == command.GOTO_WOOD:
            location = item2location[Resources.WOOD]
        elif command == command.GOTO_GRASS:
            location = item2location[Resources.GRASS]
        elif command == command.GOTO_IRON:
            location = item2location[Resources.IRON]
        elif command == command.GOTO_TOOLSHED:
            location = item2location[Tools.TOOLSHED]
        elif command == command.GOTO_WORKBENCH:
            location = item2location[Tools.WORKBENCH]
        elif command == command.GOTO_FACTORY:
            location = item2location[Tools.FACTORY]
        else:
            return

        x, y = location
        self.x = x
        self.y = y

    @property
    def encoded_theta(self):
        return self.direction.th // 90


class Cell:

    item2color = {
        Resources.WOOD: brown,
        Resources.GRASS: green,
        Resources.IRON: grey,
        Tools.TOOLSHED: dbrown,
        Tools.WORKBENCH: dgreen,
        Tools.FACTORY: dgrey
    }

    def __init__(self, config: MinecraftConfiguration, x: int, y: int, item: Optional[Item]):
        self.config = config
        self.x = x
        self.y = y
        self.item = item
        self.used = False

    def reset(self):
        self.used = False

    @property
    def encoded_item(self):
        return item2int[self.item]

    @property
    def pygame_color(self):
        return self.item2color[self.item]

    def can_get(self):
        return isinstance(self.item, Resources)

    def can_use(self):
        return isinstance(self.item, Tools)

    def draw_on_screen(self, screen):
        dx = int(self.config.offx + self.x * self.config.size_square)
        dy = int(self.config.offy + (self.config.rows - self.y - 1) * self.config.size_square)
        sqsz = (dx + 5, dy + 5, self.config.size_square - 10, self.config.size_square - 10)
        pygame.draw.rect(screen, self.pygame_color, sqsz)
        if self.used:
            pygame.draw.rect(screen, pygame.color.THECOLORS['black'],
                             (dx + 15, dy + 15, self.config.size_square - 30, self.config.size_square - 30))


class BlankCell(Cell):

    def __init__(self, config: MinecraftConfiguration, x: int, y: int):
        super().__init__(config, x, y, None)

    def draw_on_screen(self, screen):
        pass


class MinecraftGrid(PygameDrawable):

    def __init__(self, config: MinecraftConfiguration):
        self.config = config

        self.rows = config.rows
        self.columns = config.columns

        self.cells = {}  # type: Dict[Tuple[int, int], Cell]

        self._populate_token_grid()

    def _populate_token_grid(self):
        # add cells
        for t in LOCATIONS:
            x, y = t[1], t[2]
            item = t[0]  # type: Union[Resources, Tools]
            item_cell = Cell(self.config, x, y, item)
            self.cells[(x, y)] = item_cell

        # add blank cells
        for x, y in itertools.product(range(self.config.columns), range(self.config.rows)):
            if (x, y) not in self.cells:
                self.cells[(x, y)] = BlankCell(self.config, x, y)

    def reset(self):
        for t in self.cells.values():
            t.reset()

    def draw_on_screen(self, screen: pygame.Surface):
        for i in range(0, self.columns + 1):
            ox = self.config.offx + i*self.config.size_square
            pygame.draw.line(screen,
                             pygame.color.THECOLORS['black'],
                             [ox, self.config.offy],
                             [ox, self.config.offy + self.rows * self.config.size_square])

        for i in range (0, self.rows + 1):
            oy = self.config.offy + i * self.config.size_square
            pygame.draw.line(screen,
                             pygame.color.THECOLORS['black'],
                             [self.config.offx, oy],
                             [self.config.offx + self.columns * self.config.size_square, oy])

        for cell in self.cells.values():
            cell.draw_on_screen(screen)


class MinecraftState(State):

    def __init__(self, config: MinecraftConfiguration):
        self.config = config

        self.score = 0
        self.grid = MinecraftGrid(config)
        self.robot = Robot(config)

        self.last_command = NormalCommand.NOP if self.config.action_space_type == ActionSpaceType.NORMAL else\
                            DifferentialCommand.NOP if self.config.action_space_type == ActionSpaceType.DIFFERENTIAL else\
                            TeleportCommand.NOP
        self._steps = 0

        self.tasks = self.config.tasks
        self.task_progresses = OrderedDict({t.name: TaskProgress(t) for t in self.tasks})  # type: Dict[str, TaskProgress]

    def step(self, command: Union[DifferentialCommand, NormalCommand, TeleportCommand]) -> float:
        reward = 0.0
        self._steps += 1

        self.robot.step(command)
        self.last_command = command

        # handle robot outside borders
        if not (0 <= self.robot.x < self.config.columns):
            reward += self.config.reward_outside_grid
            self.robot.x = int(np.clip(self.robot.x, 0, self.config.columns - 1))
        if not (0 <= self.robot.y < self.config.rows):
            reward += self.config.reward_outside_grid
            self.robot.y = int(np.clip(self.robot.y, 0, self.config.rows - 1))

        if command == command.GET:
            reward += self._handle_get()
        elif command == command.USE:
            reward += self._handle_use()

        reward += self.config.reward_per_step
        return reward

    def reset(self) -> 'MinecraftState':
        return MinecraftState(self.config)

    def to_dict(self) -> dict:
        return {
            "x": self.robot.x,
            "y": self.robot.y,
            "theta": self.robot.encoded_theta,
            "item": self.current_cell.encoded_item,
            "command": 1 if self.last_command == self.last_command.GET else 2 if self.last_command == self.last_command.USE else 0,
            "completed_tasks": np.asarray([1 if t.is_complete() else 0 for t in self.task_progresses.values()])
        }

    @property
    def current_cell(self) -> Cell:
        return self.grid.cells[self.robot.position]

    def is_finished(self) -> bool:
        end = self._steps > self.config.horizon
        return end

    def _handle_get(self) -> float:
        reward = 0.0
        position = self.robot.x, self.robot.y
        cell = self.grid.cells[position]
        if not cell.can_get():
            reward += self.config.reward_bad_get
            return reward

        resource = cell.item
        self._do_action(Get(resource), cell)
        return reward

    def _handle_use(self) -> float:
        reward = 0.0
        position = self.robot.x, self.robot.y
        cell = self.grid.cells[position]
        if not cell.can_use():
            reward += self.config.reward_bad_use
            return reward

        tool = cell.item
        self._do_action(Use(tool), cell)
        return reward

    def _do_action(self, action: Action, cell: Cell):
        # do a step to all relevant tasks
        # if some of them completes, reset all the partial task progresses.
        some_completed = False
        for task_name, task_progress in self.task_progresses.items():
            if task_progress.is_complete(): continue
            if task_progress.is_good_action(action):
                cell.used = True
            task_progress.do_step(action)
            if task_progress.is_complete():
                some_completed = True
                break

        if some_completed:
            # reset partial tasks
            for partial_task_progress in filter(lambda t: not t.is_complete(), self.task_progresses.values()):
                partial_task_progress.reset()
            for cell in self.grid.cells.values():
                cell.reset()


class Minecraft(gym.Env, ABC):

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, configuration: Optional[MinecraftConfiguration] = None):
        self.configuration = configuration if configuration is not None else MinecraftConfiguration()
        self.state = MinecraftState(self.configuration)
        self.viewer = None  # type: Optional[PygameViewer]

    @property
    def action_space(self):
        return self.configuration.action_space

    @property
    def observation_space(self):
        return self.configuration.observation_space

    def step(self, action: int):
        command = self.configuration.get_action(action)
        reward = self.state.step(command)
        obs = self.observe(self.state)
        is_finished = self.state.is_finished()
        info = {}
        return obs, reward, is_finished, info

    def reset(self):
        self.state = MinecraftState(self.configuration)
        if self.viewer is not None:
            self.viewer.reset(self.state)
        return self.observe(self.state)

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = PygameViewer(self.state)

        return self.viewer.render(mode=mode)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

    @abstractmethod
    def observe(self, state: MinecraftState) -> gym.Space:
        """
        Extract observation from the state of the game.
        :param state: the state of the game
        :return: an instance of a gym.Space
        """

    def play(self):
        if self.configuration.action_space_type == ActionSpaceType.TELEPORT:
            raise ValueError("Play with 'teleport' action space not supported.")
        print("Press 'Q' to quit.")
        self.reset()
        self.render()
        quitted = False
        while not quitted:
            event = pygame.event.wait()
            cmd = 0
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    quitted = True
                elif event.key == pygame.K_LEFT:
                    cmd = 1
                elif event.key == pygame.K_UP:
                    cmd = 2
                elif event.key == pygame.K_RIGHT:
                    cmd = 3
                elif event.key == pygame.K_DOWN:
                    cmd = 4
                elif event.key == pygame.K_g:
                    cmd = 5
                elif event.key == pygame.K_u:
                    cmd = 6

                self.step(cmd)
                self.render()
