import numpy as np
from tqdm import tqdm
import math

import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))

from src.data.fcnet_benchmark import FCNetProteinStructureBenchmark  # noqa: E402
from src.successive_halving import successive_halving  # noqa: E402
from src.utils import plot_grey_box_optimization  # noqa: E402


def hyperband(problem, min_budget_per_model, max_budget_per_model, eta, random_seed):
    """ The hyperband algorithm

    Parameters
    ----------
    problem : instance of Problem
    min_budget_per_model : int
    max_budget_per_model : int
    eta : float
    random_seed : int

    Returns
    -------

    """
    # Todo: Compute s_max
    s_max =math.floor(math.log(max_budget_per_model,eta))
    print(s_max)
    B=(s_max+1)*max_budget_per_model
    #raise NotImplementedError()
    configs_dicts = []

    for s in tqdm(reversed(range(s_max + 1)), desc='Hyperband iter'):
        # Todo: Compute the number of models to evaluate in the HB iteration
        n_models = math.ceil((B*(eta**s))/(max_budget_per_model*(s+1)))
        #raise NotImplementedError()
        # Todo: Compute the min budget per model in the current HB iteration
        min_budget_per_model_iter = int(max_budget_per_model/eta**s)
        #raise NotImplementedError() 

        #print("s:",s,"Num of models:",n_models,"min budget:",min_budget_per_model_iter)

        configs_dict = successive_halving(problem=problem, n_models=n_models,
                                          min_budget_per_model=min_budget_per_model_iter,
                                          max_budget_per_model=max_budget_per_model, eta=eta, random_seed=random_seed)
        #print(configs_dict)
        configs_dicts.append(configs_dict)
        #print(configs_dicts)

    return configs_dicts


if __name__ == '__main__':
    problem = FCNetProteinStructureBenchmark(data_dir="C:/Users/Vineet/Desktop/mission future/Auto_ML/Week7_Speedup_Techniques_For_HPO/Assignment/ex07-grey-box/src/data/fcnet_tabular_benchmarks/fcnet_tabular_benchmarks/")
    configs_dicts = hyperband(problem=problem, eta=2, random_seed=0, max_budget_per_model=100,
                              min_budget_per_model=2)
    plot_grey_box_optimization(configs_dicts, min_budget_per_model=2)
