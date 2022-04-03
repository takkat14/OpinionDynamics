from mesa import Model
from mesa.datacollection import DataCollector
from agent import OpinionAgent
from scheduler import SmartInteractionStagedActivation


class OpinionModel(Model):
    def __init__(self, N, mu, init_u, extremist_ratio=None, d=None, extremist_u=None, regime="p2p"):
        super().__init__()
        self.num_agents = N
        self.num_extremist = int(self.num_agents * extremist_ratio)
        self.d = d  # d * self.num_extremist = p+ - p- => p+ = (d + 1) * self.ext / 2
        self.p_plus = (self.d + 1) * self.num_extremist / 2
        self.init_u = init_u
        self.u_e = extremist_u
        # TODO: add a distribution for the parameter 'mu', since it is the coefficient of impact on a certain person
        self.mu = mu
        self.regime = regime
        self.stage_list = ["fetch_" + self.regime, "apply"]
        self.schedule = SmartInteractionStagedActivation(self, self.stage_list, regime=self.regime)
        # Create agents
        opinions = [self.random.uniform(-1, 1) for _ in range(self.num_agents)]  # Красота вышла из чата
        opinions.sort()
        for i, x in enumerate(opinions):
            # TODO: add some regimes of uncertainty sampling
            u = self.u_e if (self.num_agents - self.p_plus <= i) or (i < self.num_extremist - self.p_plus) else self.init_u
            a = OpinionAgent(i, self, x, u, self.mu, model_regime=self.regime)
            self.schedule.add(a)
        self.datacollector = DataCollector(
            # TODO: add convergence checker over
            # model_reporters={"Convergence": check_convergence},
            agent_reporters={"Opinion": "x", "Uncertainty": "u", "Delta X": "delta_x", "Delta U": "delta_u"})

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
