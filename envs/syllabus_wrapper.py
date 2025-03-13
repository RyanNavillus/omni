import numpy as np
from syllabus.core import TaskWrapper
from syllabus.task_space import DiscreteTaskSpace, StratifiedDiscreteTaskSpace
from crafter import objects, worldgen


class CrafterTaskWrapper(TaskWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.task_space = DiscreteTaskSpace(len(self.env.given_achievements), list(self.env.given_achievements.keys()))

        # strata = []
        # strata_idx = []
        # stratum = []
        # stratum_idx = []
        # for idx, task in enumerate(self.env.given_achievements.keys()):
        #     if len(stratum) == 0:
        #         stratum.append(task)
        #         stratum_idx.append(idx)
        #     else:
        #         if task.startswith(stratum[0]):
        #             stratum.append(task)
        #             stratum_idx.append(idx)
        #         else:
        #             strata.append(stratum)
        #             strata_idx.append(stratum_idx)
        #             stratum = [task]
        #             stratum_idx = [idx]

        # self.task_space = StratifiedDiscreteTaskSpace(strata_idx, strata)
        self.task = self.task_space.decode(self.env.task_idx)

    def reset(self, *args, **kwargs):
        new_task = kwargs.pop("new_task", None)
        options = kwargs.pop("options", None)

        obs, info = self.env.reset()
        if new_task is not None:
            self.change_task(new_task)
        elif options is not None:
            self.change_task(options)
        else:
            self.change_task(self.task)
        obs["task_enc"] = self.env.task_enc
        info["task_completion"] = self.env.task_progress
        info["task"] = self.env.task_idx
        return obs, info

    def change_task(self, new_task):
        self.task = new_task
        if isinstance(new_task, str):
            new_task = self.task_space.encode(new_task)
        self.env.task_idx = new_task
        self.env.task_steps = 0
        self.env.task_enc = self.env._encode_task(self.env.task_idx)
        self.env.update_given_ach()
        self.env.task_progress = 0

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        info["task_completion"] = self.env.task_progress
        info["task"] = self.env.task_idx

        return obs, reward, term, trunc, info


class CrafterSeedWrapper(TaskWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.task_space = DiscreteTaskSpace(100)
        self.task = self.env._seed
        self.assigned_tasks = 0
        self.episode_return = 0

    def reset(self, *args, **kwargs):
        self.assigned_tasks = 0
        self.episode_return = 0

        new_task = kwargs.pop("new_task", None)
        options = kwargs.pop("options", None)

        if new_task is not None:
            self.change_task(new_task)
        elif options is not None:
            self.change_task(options)
        else:
            self.change_task(self.env._seed)
        obs, info = self.env.reset()
        obs["task_enc"] = self.env.task_enc
        info["task_completion"] = self.env.task_progress
        info["task"] = self.task
        return obs, info

    def change_task(self, new_task):
        self.task = new_task
        self.env.seed(self.task)

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        self.episode_return += reward
        info["task_completion"] = self.env.task_progress
        info["task"] = self.task

        return obs, reward, term, trunc, info
