from mesa import Model
from mesa.datacollection import DataCollector
from model.agent import OpinionAgent
import numpy as np
from model.scheduler import SmartInteractionStagedActivation


class OpinionModel(Model):
    def __init__(self, num_agents, mu, init_u, extremist_ratio=0,
                 d=0, extremist_u=0.05, regime="p2p", eps=1e-6, seed=42,
                 max_iter=1000):
        super().__init__()
        self.num_agents = num_agents
        self.num_extremist = int(self.num_agents * extremist_ratio)
        self.d = d  # d * self.num_extremist = p+ - p- => p+ = (d + 1) * self.ext / 2
        self.p_plus = int((self.d + 1) * self.num_extremist / 2)
        self.init_u = init_u
        self.u_e = extremist_u
        self.mu = mu
        self.regime = regime
        self.stage_list = ["fetch_" + self.regime, "apply"]
        self.schedule = SmartInteractionStagedActivation(self, self.stage_list, regime=self.regime)
        self.eps = eps
        self.random.seed(seed)
        self.max_iter = max_iter

        # Create agents
        opinions = [self.random.uniform(-1, 1) for _ in range(self.num_agents)]  # Красота вышла из чата
        opinions.sort()
        for i, x in enumerate(opinions):
            if (self.num_agents - self.p_plus <= i) or (i < self.num_extremist - self.p_plus):
                u = self.u_e
                is_extremist = True
            else:
                u = self.init_u
                is_extremist = False
            a = OpinionAgent(i, self, x, u, self.mu, model_regime=self.regime, is_extremist=is_extremist)
            self.schedule.add(a)
        self.datacollector = DataCollector(
            model_reporters={"Y": self.check_clusters_convergence},
            agent_reporters={"Opinion": "x", "Uncertainty": "u", "Delta X": "historic_delta_x",
                             "Delta U": "historic_delta_u"})

    def check_convergence(self) -> bool:
        for agent in self.schedule.agents:
            if abs(agent.historic_delta_x) > self.eps or abs(agent.historic_delta_u) > self.eps:
                return False
        return True

    def run_model(self) -> None:
        while self.running:
            if self.schedule.steps % 100 == 0:
                print(f"Elapsed {self.schedule.steps} steps")
            self.step()
            if self.check_convergence():
                self.running = False
            elif self.schedule.steps >= self.max_iter:
                self.running = False
                raise BaseException("Model doesn't converge")

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

    def check_clusters_convergence(self):
        became_positive = 0
        became_negative = 0
        for agent in self.schedule.agents:
            if not agent.is_extremist:
                if -1 <= round(agent.x, 1) <= -0.9:
                    became_negative += 1
                elif 0.9 <= round(agent.x, 1) <= 1:
                    became_positive += 1
        num_moderate = self.num_agents - self.num_extremist
        y = (became_positive / num_moderate) ** 2 + (became_negative / num_moderate) ** 2
        return y

    def get_clusters(self):
        clusters = np.round(np.array(self.datacollector._agent_records[self.schedule.steps - 1])[:, 2], 1)
        clusters_count = {cluster: 0 for cluster in set(clusters)}
        for opinion in clusters:
            clusters_count[opinion] += 1
        return clusters_count
