from syllabus.core import TaskWrapper
from syllabus.task_space import DiscreteTaskSpace


class CrafterTaskWrapper(TaskWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.task_space = DiscreteTaskSpace(1128)

    def reset(self, *args, **kwargs):
        new_task = kwargs.pop("new_task", None)
        obs, info = self.env.reset()
        if new_task is not None:
            self.change_task(new_task)
        return obs, info

    def change_task(self, new_task):
        self.task = new_task
        self.env.task_idx = new_task
        self.env.task_steps = 0
        self.env.task_enc = self.env._encode_task(self.env.task_idx)
        self.env.update_given_ach()
        self.env.task_progress = 0

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        info["task_completion"] = self.env.task_progress
        return obs, reward, term, trunc, info
