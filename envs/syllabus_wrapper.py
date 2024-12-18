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
        self.task_enc = self.env._encode_task(self.env.task_idx)
        obs, info = self.env.reset()
        return obs, info
