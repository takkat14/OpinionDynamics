from mesa import Model
from mesa.datacollection import DataCollector
from agent import OpinionAgent
from scheduler import SmartInteractionStagedActivation


class OpinionModel(Model):
    def __init__(self, N, mu, init_u, extremist_ratio=None, d=None, extremist_u=None, regime="p2p", eps=1e-6, seed=42):
        super().__init__()
        self.num_agents = N
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
        # Create agents
        opinions = [self.random.uniform(-1, 1) for _ in range(self.num_agents)]  # Красота вышла из чата
        opinions.sort()
        for i, x in enumerate(opinions):
            # TODO: add some regimes of uncertainty sampling
            if (self.num_agents - self.p_plus <= i) or (i < self.num_extremist - self.p_plus):
                u = self.u_e
                is_extremist = True
            else:
                u = self.init_u
                is_extremist = False
            a = OpinionAgent(i, self, x, u, self.mu, model_regime=self.regime, is_extremist=is_extremist)
            self.schedule.add(a)
        self.datacollector = DataCollector(
            agent_reporters={"Opinion": "x", "Uncertainty": "u", "Delta X": "delta_x", "Delta U": "delta_u"})

    def check_convergence(self) -> bool:
        for i, agent in self.schedule._agents.items():
            if abs(agent.delta_x) > self.eps or abs(agent.delta_u) > self.eps:
                return False
        return True

    def run_model(self) -> None:
        while self.running:
            self.step()
            if self.check_convergence():
                self.running = False

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
