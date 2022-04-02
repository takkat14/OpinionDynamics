from mesa import Agent
import numpy as np


# Only regime p2p
class OpinionAgent(Agent):
    """ Agent impl"""

    def __init__(self, unique_id, model, x, u, mu, model_regime="p2p"):
        super().__init__(unique_id, model)
        self.x = x
        self.u = u
        self.mu = mu
        self.delta_x = 0
        self.delta_u = 0
        # self.steps = 0
        self.regime = model_regime

    def fetch_p2p(self, other_agent):
        overlap = min(self.x + self.u, other_agent.x + other_agent.u) - max(self.x - self.u,
                                                                            other_agent.x - other_agent.u)
        delta_us = overlap / self.u - 1
        delta_other = overlap / other_agent.u - 1
        if delta_us > 0:
            self.delta_x += self.mu * delta_us * (other_agent.x - self.x)
            self.delta_u += self.mu * delta_us * (other_agent.u - self.u)
        if delta_other > 0:
            other_agent.delta_x += other_agent.mu * delta_other * (self.x - other_agent.x)
            other_agent.delta_u += other_agent.mu * delta_other * (self.u - other_agent.u)

    def apply(self):
        self.x += self.delta_x
        self.x = max(self.x, -1)
        self.x = min(self.x, 1)
        self.u += self.delta_u
        self.delta_x = 0
        self.delta_u = 0

    # def step(self):
    #     # do something with influence
    #     self.steps += 1
