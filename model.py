from mesa import Model
from mesa.datacollection import DataCollector
from agent import OpinionAgent
from scheduler import SmartInteractionStagedActivation


class OpinionModel(Model):
    def __init__(self, N, mu, init_u, d=None, regime="p2p"):
        """

        :param N: int -- number of agents in model
        :param mu: float -- constant parameter which amplitude controls the speed of the dynamics
        :param init_u: float -- uncertainty value for moderators (until sampling is not implemented)
        :param d: float -- extremists ratio
        :param regime: str -- "p2p" or "all" -- agents' interaction regime
        """
        super().__init__()
        self.num_agents = N  # TODO: add extremists
        self.d = d
        self.init_u = init_u
        # TODO: add a distribution for the parameter 'mu', since it is the coefficient of impact on a certain person
        self.mu = mu
        self.regime = regime
        self.stage_list = ["fetch_" + self.regime, "apply"]
        self.schedule = SmartInteractionStagedActivation(self, self.stage_list, regime=self.regime)
        # Create agents
        for i in range(self.num_agents):
            x = self.random.uniform(-1.01, 1.01)
            u = self.init_u  # TODO: add some regimes of uncertainty sampling
            a = OpinionAgent(i, self, x, u, self.mu, model_regime=self.regime)
            self.schedule.add(a)
        self.datacollector = DataCollector(
            # TODO: add convergence checker
            # model_reporters={"Convergence": check_convergence},
            agent_reporters={"Opinion": "x", "Uncertainty": "u", "Delta X": "delta_x", "Delta U": "delta_u"})

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
