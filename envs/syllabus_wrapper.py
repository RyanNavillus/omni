from syllabus.core import TaskWrapper
from syllabus.task_space import DiscreteTaskSpace


class CrafterTaskWrapper(TaskWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.task_space = DiscreteTaskSpace(1128)

    def reset(self, *args, **kwargs):
        if "new_task" in kwargs:
            new_task = kwargs.pop("new_task")
            self.task = new_task
            self.change_task(new_task)
        obs, info = self.env.reset()
        return obs, info

    def change_task(self, new_task):
        self.env.task_idx = new_task
        self.env.task_steps = 0
        self.env.task_enc = self.env._encode_task(self.task_idx)
