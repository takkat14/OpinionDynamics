from mesa.time import BaseScheduler


# Тут мы реализуем свой fancy шедулер,
# чтобы воспроизвести логику заморозки времени во время взаимодействий агентов


class SmartInteractionStagedActivation(BaseScheduler):
    def __init__(self,
                 model,
                 stage_list,
                 shuffle: bool = False,
                 shuffle_between_stages: bool = False,
                 regime="p2p"):
        """
        Custom scheduler class for Deffuant model
        !Every agent should have method "fetch" and "apply"!

        :param model: mesa.Model -- model of scheduler
        :param stage_list: List[str] -- list of stages in execution order
        :param shuffle: bool -- shuffle agents before step
        :param shuffle_between_stages: bool -- shuffle agents between stages
        :param regime: str -- "p2p" or "all" -- agents' interaction regime
        """
        super().__init__(model)
        # Stage_list -- список функций в агенте,
        # которые нужно вызвать для каждого агента последовательно.
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
            # Ну, до оптимальных времен
            for agent_key in agent_keys:
                if "fetch" in stage:
                    if self.regime == "p2p":
                        # Пока что, тут есть риск убедить самого себя
                        # В какой-то мере, это логично, так что исправлять не буду.
                        # Выбираем агента, с кем нужно повзаимодействовать
                        other_agent = self.model.random.choice(agent_keys)
                        if other_agent == agent_key:
                            other_agent = self.model.random.choice(agent_keys)
                        getattr(self._agents[agent_key], stage)(self._agents[other_agent])  # Запускаем функцию.
                    if self.regime == "all":
                        getattr(self._agents[agent_key], stage)()
                elif "apply" in stage:
                    getattr(self._agents[agent_key], stage)()  # Применяем все изменения для нашего малышарика.
                else:
                    raise AttributeError("Not only fetch or apply in stage list")
            if self.shuffle_between_stages:
                self.model.random.shuffle(agent_keys)
            self.time += self.stage_time
        self.steps += 1
