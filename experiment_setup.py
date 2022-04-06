from model.model import OpinionModel
from mesa.batchrunner import batch_run

class Experiment:
    def __init__(self, model_cls, params):
        self.model_cls = model_cls
        self.params = params

    def run_experiment(self):
        batch_run(self.model_cls, )