from copy import deepcopy
import re
import time
import gymnasium as gym
import numpy as np
from PIL import Image, ImageFont, ImageDraw

from crafter import constants
from crafter import engine
from crafter import objects
from crafter import worldgen
from crafter import env
from envs.env_utils import get_repeat_tasks

DiscreteSpace = gym.spaces.Discrete
BoxSpace = gym.spaces.Box
DictSpace = gym.spaces.Dict
BaseClass = gym.Env


OBJS = ['table', 'furnace']
MATS = ['wood', 'stone', 'coal', 'iron', 'diamond', 'drink']
TOOLS = ['sword', 'pickaxe']
INSTRS = ['collect', 'make', 'place']
COUNT = [str(i) for i in range(2, 11)]
ENC_ORDER = INSTRS + OBJS + MATS + TOOLS + COUNT
DUMMY_BITS = 10  # for 2^N dummy tasks, min=1
DUMMY_TASKS = np.power(2, DUMMY_BITS) - 1


class Env(env.Env):

    def __init__(
            self, area=(64, 64), view=(9, 9), size=(64, 64),
            reward=True, length=1500, seed=None, eval_mode=False, dummy_bits=10, static_task="make_wood_pickaxe", num_worlds=100, **kwargs):
        # TODO: Remove Hack
        global DUMMY_BITS, DUMMY_TASKS
        DUMMY_BITS = dummy_bits
        DUMMY_TASKS = np.power(2, dummy_bits) - 1

        super().__init__(area, view, size, reward, length, seed, **kwargs)
        self.eval_mode = eval_mode
        self.static_task = static_task
        self.num_worlds = num_worlds
        counts = [10 if 'collect' in ach else 5 for ach in constants.achievements]
        self.target_achievements, self.isreptask = get_repeat_tasks(constants.achievements, counts=counts)
        self.task_progress = 0
        # task condition attributes
        self.task_idx = self.target_achievements.index(self.static_task)
        self.task_enc = np.zeros(len(ENC_ORDER) + DUMMY_BITS, dtype=np.uint8)
        self.past_achievements = None
        self.follow_achievements = {k: 0 for k in self.target_achievements}
        self.given_achievements = {k: 0 for k in self.target_achievements}
        self.given_achievements.update({f'dummy{i}': 0 for i in range(DUMMY_TASKS)})
        self.task_enc = self._encode_task(self.task_idx)
        self.eval_tsr = np.zeros(len(self.target_achievements))  # evaluated task success rates

        # Create spaces
        self._action_space = DiscreteSpace(len(constants.actions))
        img_shape = (self._size[1], self._size[0], 3)
        self._observation_space = DictSpace({
            'image': BoxSpace(0, 255, img_shape, np.uint8),
            'task_enc': BoxSpace(0, 1, (len(ENC_ORDER) + DUMMY_BITS, ), np.uint8),
        })
        self._center = (self._area[0] // 2, self._area[1] // 2)
    #     self.generate_worlds()

    # def generate_worlds(self):
    #     start = time.time()
    #     self._worlds = []

    #     for i in range(self.num_worlds):
    #         _world = engine.World(self._area, constants.materials, (12, 12))
    #         _world.reset(seed=i)
    #         _player = objects.Player(self._world, self._center)
    #         _world.add(_player)
    #         worldgen.generate_world_og(_world, _player)
    #         _world.remove(_player)
    #         self._worlds.append(_world)
    #     end = time.time()
    #     print("World gen time: ", end - start)

    def reset_player(self):
        self._player.inventory = {'health': 0}
        self._player.removed = False
        self._player.pos = self._center
        self._player.facing = (0, 1)
        self._player.inventory = {name: info['initial'] for name, info in constants.items.items()}
        self._player.achievements = {name: 0 for name in constants.achievements}
        # self._player.action = 'noop'
        # self._player.sleeping = False
        # self._player._hunger = 0
        # self._player._thirst = 0
        # self._player._fatigue = 0
        # self._player._recover = 0
        # self._player.achs_type = ""

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def seed(self, seed=0):
        self._seed = seed

    def reset(self):
        inventory = None
        # inherit inventory 50% of the time
        if not self.eval_mode and self._player and np.random.rand() < 0.5:
            inventory = self._player.inventory.copy()

        center = (self._world.area[0] // 2, self._world.area[1] // 2)
        self._episode += 1
        self._step = 0
        self._world.reset(seed=hash((self._seed, self._episode)) % (2 ** 31 - 1))
        self._update_time()
        self._player = objects.Player(self._world, center)
        self._world.add(self._player)
        self._unlocked = set()
        worldgen.generate_world_og(self._world, self._player)
        # self._unlocked = set()
        # self._episode += 1
        # self._step = 0

        # # Reset world to initial state
        # template_world, template_player = self._worlds[self._seed % self.num_worlds]
        # self._world = deepcopy(template_world)
        # self._player = deepcopy(template_player)
        # for obj in self._world.objects:
        #     obj.world = self._world
        # self._player.world = self._world
        # self._world.add(self._player)
        # self._world.random.seed(self._seed)
        # self._update_time()

        if inventory:
            self._player.inventory = inventory
        self.past_achievements = self._player.achievements.copy()
        self.follow_achievements = {k: 0 for k in self.target_achievements}
        self.given_achievements = {k: 0 for k in self.target_achievements}
        self.given_achievements.update({f'dummy{i}': 0 for i in range(DUMMY_TASKS)})
        return self._obs(), {}

    def update_given_ach(self):
        if self.task_idx < len(self.target_achievements):
            self.given_achievements[self.target_achievements[self.task_idx]] += 1
        else:
            i = self.task_idx - len(self.target_achievements)
            self.given_achievements[f'dummy{i}'] += 1

    def step(self, action):
        obs, reward, done, other_done, info = super().step(action)
        # additional info
        info['given_achs'] = self.given_achievements.copy()
        info['follow_achs'] = self.follow_achievements.copy()
        if done and self.task_progress < 1.0:
            # task failed
            self.task_progress = -1.0
        elif self.task_progress >= 1.0:
            done = True
        return obs, reward, done, other_done, info

    def _encode_task(self, task_idx):
        encoding = np.zeros(len(ENC_ORDER) + DUMMY_BITS, dtype=np.uint8)
        if self.task_idx < len(self.target_achievements):
            task = self.target_achievements[self.task_idx]
            task_words = task.split('_')
            # bag of words encoding
            for i, word in enumerate(ENC_ORDER):
                if word in task_words:
                    encoding[i] = 1
        else:
            dummy_enc = np.random.choice([0, 1], size=DUMMY_BITS)
            encoding[-DUMMY_BITS:] = dummy_enc
            # ensure that there is at least one bit flipped in dummy bits
            rdn_idx = np.random.randint(DUMMY_BITS, size=1)
            encoding[-rdn_idx-1] = 1
        return encoding

    def _decode_task(self, task_enc):
        if (task_enc[-DUMMY_BITS:] > 0).any():
            return 'dummy task'
        else:
            return ' '.join([ENC_ORDER[int(i)] for i, c in enumerate(task_enc) if c])

    def _specify_curri_task(self):
        # choose random next task
        pass

    def _specify_eval_task(self):
        # choose next task
        # NOTE: no need to eval dummy tasks
        pass

    def render(self, size=None, semantic=False, add_desc=False):
        canvas = super().render(size=size, semantic=semantic)
        if not semantic and add_desc:
            img = Image.fromarray(canvas)
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 25)
            draw.text((0, 0), self._decode_task(self.task_enc), (255, 255, 255), font=font)
            # draw.text((0, 20), self._player.action,(255,255,255), font=font)
            canvas = np.asarray(img)
        return canvas

    def _get_reward(self):
        reward = 0

        unlocked = {
            name for name, count in self._player.achievements.items()
            if count > 0 and name not in self._unlocked}
        self._unlocked |= unlocked

        if self.task_idx < len(self.target_achievements):
            task_desc = self.target_achievements[self.task_idx]
            if self.isreptask[self.task_idx]:
                # repeat task
                task_words = task_desc.split('_')
                subtask_desc = '_'.join(task_words[:-1])
                if self._player.achievements[subtask_desc] - self.past_achievements[subtask_desc] > 0:
                    self.task_progress += 1.0 / float(task_words[-1])
            elif self._player.achievements[task_desc] - self.past_achievements[task_desc] > 0:
                # agent successfully completed given task
                self.task_progress = 1.0

            # Set reward and achievements
            if self.task_progress >= 1.0:
                reward += 1.0
                self.follow_achievements[task_desc] += 1

        self.past_achievements = self._player.achievements.copy()
        return reward

    def _obs(self):
        return {
            'image': self.render(),
            'task_enc': self.task_enc,
        }

    def set_curriculum(self, train=False):
        pass

    def push_info(self, info):
        self.eval_tsr = info['ema_tsr']
