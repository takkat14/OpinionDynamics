from mesa.time import BaseScheduler
import numpy as np


class SmartInteractionStagedActivation(BaseScheduler):
    """
    Every agent should have method "fetch" and "apply"
    """

    def __init__(self,
                 model,
                 stage_list,
                 shuffle: bool = False,
                 shuffle_between_stages: bool = False,
                 regime="p2p"):
        super().__init__(model)
        self.stage_list = stage_list
        self.shuffle = shuffle
        self.shuffle_between_stages = shuffle_between_stages
        self.stage_time = 1 / len(self.stage_list)
        self.regime = regime

    def step(self) -> None:
        """Executes all the stages for all agents."""
        agent_keys = list(self._agents.keys())
        if self.shuffle:
            self.model.random.shuffle(agent_keys)
        for stage in self.stage_list:
            for i, agent_key in enumerate(agent_keys):
                if "fetch" in stage:
                    if self.regime == "p2p":
                        # Пока что, тут есть риск убедить самого себя
                        # В какой-то мере, это логично, так что исправлять не буду
                        other_agent = self.model.random.choice(agent_keys)  # Sample other agent
                        getattr(self._agents[agent_key], stage)(self._agents[other_agent])  # Run stage
                elif "apply" == stage:
                    getattr(self._agents[agent_key], stage)()
            if self.shuffle_between_stages:
                self.model.random.shuffle(agent_keys)
            self.time += self.stage_time

        self.steps += 1
