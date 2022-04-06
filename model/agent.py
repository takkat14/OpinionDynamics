from mesa import Agent


class OpinionAgent(Agent):
    def __init__(self, unique_id, model, x, u, mu, model_regime="p2p", is_extremist=False):
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
        self.historic_delta_x = 0
        self.historic_delta_u = 0
        self.regime = model_regime
        if self.regime == "all":
            self.mu /= self.model.num_agents
        self.is_extremist = is_extremist

    def fetch_p2p(self, other_agent):
        overlap = min(self.x + self.u, other_agent.x + other_agent.u) - max(self.x - self.u,
                                                                            other_agent.x - other_agent.u)
        if overlap > other_agent.u:
            delta_us = overlap / other_agent.u - 1
            self.delta_x += self.mu * delta_us * (other_agent.x - self.x)
            self.delta_u += self.mu * delta_us * (other_agent.u - self.u)

    def fetch_all(self):
        for agent in self.model.schedule.agents:
            if agent.unique_id != self.unique_id:
                self.fetch_p2p(agent)

    def apply(self):
        self.x += self.delta_x
        self.x = max(self.x, -1)
        self.x = min(self.x, 1)
        self.u += self.delta_u
        self.u = max(self.u, 0)
        self.u = min(self.u, 2)
        self.historic_delta_x = self.delta_x
        self.historic_delta_u = self.delta_u
        self.delta_x = 0
        self.delta_u = 0
