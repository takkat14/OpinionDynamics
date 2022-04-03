from mesa import Agent


class OpinionAgent(Agent):
    def __init__(self, unique_id, model, x, u, mu, model_regime="p2p"):
        """
        Implementation of agent in Deffuant model

        :param unique_id: int -- unique identification of an agent
        :param model: mesa.Model -- model for an agent
        :param x: float -- initial opinion of an agent
        :param u: float -- uncertainty level of an agent
        :param mu: float -- constant parameter which amplitude controls the speed of the dynamics
        :param model_regime: str -- "p2p" or "all" -- agents' interaction regime
        """
        super().__init__(unique_id, model)
        self.x = x
        self.u = u
        self.mu = mu
        self.delta_x = 0
        self.delta_u = 0
        # self.steps = 0
        self.regime = model_regime

    def fetch_p2p(self, other_agent):
        self.delta_x = 0
        self.delta_u = 0

        overlap = min(self.x + self.u, other_agent.x + other_agent.u) - max(self.x - self.u,
                                                                            other_agent.x - other_agent.u)
        if overlap > other_agent.u:
            delta_us = overlap / other_agent.u - 1
            self.delta_x += self.mu * delta_us * (other_agent.x - self.x)
            self.delta_u += self.mu * delta_us * (other_agent.u - self.u)
        if overlap > self.u:
            delta_other = overlap / self.u - 1
            other_agent.delta_x += other_agent.mu * delta_other * (self.x - other_agent.x)
            other_agent.delta_u += other_agent.mu * delta_other * (self.u - other_agent.u)

    def fetch_all(self):
        # Неоптимально, потому что мозги вышли из чата
        for agent in self.model.schedule._agents:
            self.fetch_p2p(self.model.schedule._agents[agent])

    def apply(self):
        self.x += self.delta_x
        self.x = max(self.x, -1)
        self.x = min(self.x, 1)
        self.u += self.delta_u
        self.u = max(self.u, 1e-10)
        self.u = min(self.u, 2)
