from typing import Optional
import numpy as np
from pandas import DataFrame


def apriori(costs: np.ndarray, weights: Optional = None, order: Optional = None):
    """
    Implement the two example apriori methods presented in the lecture
    Parameters
    ----------
    costs   (n_points, m_costs) array
    weights (m_costs, ) array. Determines the weighting of the costs. If None use lexical ordering
    order   (m_costs, ) array. Determines the lexicograpical order. If None use weighted sum

    Returns
    -------
    Index of optimal element according to apriori method
    """
    if weights is not None and order is not None:
        raise Exception('You can only specify weight or order but not both')
    if weights:
        #raise NotImplementedError
        lst=[]
        leng=len(costs[0])
        #print("debug1:",leng)
        for i in costs:
            var=0
            for j in range(leng):
                var=var+i[j]*weights[j]
            lst.append(var)
        #print(lst)
        arg=np.argmin(lst)        
    if order:
        leng=len(costs[0])
        lst1=costs.copy()
        print(lst1)
        for i in range (leng):
            for j in order:
                if(i==j):
                    ind=order.index(i)
            print("index",ind)
            lst=[]
            for k in lst1:
                lst.append(k[ind])
            print(lst)
            min_val=np.min(lst)
            print("min",min_val)
            lst2=[]
            for h in lst1:
                if (h[ind]==min_val):
                    lst2.append(h)
            print(lst2)
            lst1=lst2.copy()
        #raise NotImplementedError
        print(lst1[0])
        for m in range (len(costs)):
            if ((costs[m]==lst1[0]).all()):
                arg=m
                break
        print(costs[80][12])
    return arg
