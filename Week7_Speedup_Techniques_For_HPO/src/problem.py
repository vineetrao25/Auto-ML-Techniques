from abc import abstractmethod, ABC


class Problem(ABC):
    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def objective_function(self, config, budget):
        pass

    @abstractmethod
    def get_configuration_space(self):
        pass

    @abstractmethod
    def get_results(self):
        pass
