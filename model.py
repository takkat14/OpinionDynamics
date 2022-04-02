from mesa import Model
from agent import OpinionAgent
from scheduler import SmartInteractionStagedActivation


class OpinionModel(Model):
    def __init__(self, N, mu, init_u, regime="p2p"):
        self.num_agents = N
        self.mu = mu
        self.regime = regime
        self.stage_list = ["fetch_" + self.regime, "apply"]
        self.schedule = SmartInteractionStagedActivation(self, self.stage_list, regime=self.regime)
        # Create agents
        for i in range(self.num_agents):
            x = self.random.uniform(-1, 1)
            a = OpinionAgent(i, self, x, init_u, self.mu, model_regime=self.regime)
            self.schedule.add(a)
            # Add the agent to a random grid cell

        # self.datacollector = DataCollector(
        #     model_reporters={"Gini": compute_gini},
        #     agent_reporters={"Wealth": "wealth"}
        # )

    def step(self):
        # self.datacollector.collect(self)
        self.schedule.step()
