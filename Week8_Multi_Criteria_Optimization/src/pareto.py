from os import lstat
import numpy as np
from numpy.core import fromnumeric
from numpy.lib.function_base import kaiser


def pareto(costs: np.ndarray):
    """
    Find the pareto-optimal points
    :param costs: (n_points, m_cost_values) array
    :return: (n_points, 1) boolean array indicating if point is on pareto front or not.
    """
    # first assume all points are pareto optimal
    is_pareto = np.ones(costs.shape[0], dtype=bool)
    #print(is_pareto)
    for i,c in enumerate (costs):
        if is_pareto[i]:
            is_pareto[is_pareto]=np.any(costs[is_pareto]<c, axis=1)
            is_pareto[i]=True

    # Stepwise eliminate all elements that are dominated by any other point
    #raise NotImplementedError
    #print(is_pareto)
    return is_pareto


def nDS(costs):
    """
    Implementation of the non-dominated sorting method
    :param costs: (n_points, m_cost_values) array
    :return: list of all fronts
    """
    # Stepwise compute the pareto front without all prior dominating points
    tmp_lst=costs.copy()
    fronts=[]
    tmp1=[]
    tmp3=[]
    #print(tmp_lst)
    
    while(len(tmp_lst)>1):
    
        tmp1=pareto(tmp_lst)
        tmp2=np.argwhere(tmp1).flatten().tolist()
        #print("index of front:",tmp2)
        #print()
        
        
        tmp_lst=tmp_lst.tolist()
        tmp4=[tmp_lst[k] for k in tmp2]
        #print(tmp4)
        fronts.append(tmp4)
        #print("final front",fronts)

        tmp_lst=[tmp_lst[e] for e in range (len(tmp_lst)) if e not in tmp2]
        tmp_lst=np.array(tmp_lst)
        #print(len(tmp_lst))
    
    fronts.append(tmp_lst.tolist())
    #print(fronts)
    fronts=np.array(fronts)
    fronts=fronts.reshape(-1,1)



    #raise NotImplementedError
    return fronts


def crowdingDist(front):
    """
    Implementation of the crowding distance
    :param front: (n_points, m_cost_values) array
    :return: sorted_front and corresponding distance value of each element in the sorted_front
    """
    # TODO
    # first sort the front (first sort the first column then the second, third, ...)
    print(front)
    sorted_front=[]
    front1=front.copy()
    front=front.tolist()
    dists=np.zeros(len(front))
    for l in range(len(front[0])):
        sorted_front=sorted(front,key=lambda x:x[l],reverse=True)
        print("SOTRED FRONT:",l,sorted_front)
        lst=[]
        lst1=[]
        for i in sorted_front:
            lst1.append(front.index(i))
            lst.append(i[l])
        #print(lst)
        print(lst1)
        lst_min,lst_max=min(lst),max(lst)
        leng=len(front)
        for j ,val in enumerate(lst):
            lst[j]=(val-lst_min)/(lst_max-lst_min)
        for k in range (1,leng-1):
            term1=abs(lst[k+1]-lst[k-1])
            #print(term1)
            #term2=lst_max-lst_min
            #print(term2)
            dists[k]=term1
        dists[0]=np.inf
        dists[leng-1]=np.inf
        #print(dists)
        #ists1=[x for _, x in sorted(zip(lst1, dists), key=lambda pair: pair[0])]
        #dists=dists1.copy()
        print("DISTS:",l,dists)
    
    #print(sorted_front)
    #print(dists)
    sorted_front=np.array(sorted_front)
    # TODO
    # on the sorted front compute the distance of all points to its neighbor    raise NotImplementedError

    return sorted_front, dists


def computeHV2D(front, ref):
    """
    Compute the Hypervolume for the pareto front  (only implement it for 2D)
    :param front: (n_points, m_cost_values) array for which to compute the volume
    :param ref: coordinates of the reference point
    :returns: Hypervolume of the polygon spanned by all points in the front + the reference point
    """
    # TODO
    # sort front to get "outline" of polygon (don't forget to add the reference point to that outline)

    # TODO
    # You can use the shoelace formula to compute area as we constrain ourselves to 2D
    # (https://en.wikipedia.org/wiki/Shoelace_formula)
    return None


if __name__ == '__main__':
    # We prepared some plotting code for you to check your pareto front implementation and the non-dominating sorting
    from sklearn.datasets import load_boston
    from matplotlib import pyplot as plt
    import seaborn as sb
    sb.set_style('darkgrid')

    wine = load_wine()
    X = wine['data']  # 1, 2, 6 as features
    # metric contains "malic acid", "ash", "nonflavanoid phenols"
    costs2D = X[:, [1, 2]]
    costs3D = X[:, [1, 2, 6]]

    plt.scatter(costs2D[:, 0], costs2D[:, 1])
    pareto_front = costs2D[pareto(costs2D)]

    # sort for plotting
    pareto_front = np.sort(pareto_front.view([('', pareto_front.dtype)] * pareto_front.shape[1]),
                           order=['f1'],  # order by first element then second and so on
                           axis=0).view(np.float)
    plt.scatter(pareto_front[:, 0], pareto_front[:, 1], marker='.', c='orange')
    plt.plot(pareto_front[:, 0], pareto_front[:, 1], marker='.', c='orange')
    plt.title('Pareto Front on partial Wine Dataset')
    plt.xlabel('malic acid')
    plt.ylabel('ash')
    plt.show()

    # metric contains malic acid and nonflavoid phenoms
    costs2D = X[:, [1, 6]]

    plt.scatter(costs2D[:, 0], costs2D[:, 1])

    fronts = nDS(costs2D)
    for pareto_front in fronts:
        # sort for plotting
        pareto_front = np.sort(pareto_front.view([('', pareto_front.dtype)] * pareto_front.shape[1]),
                               order=['f1'],  # order by first element then second and so on
                               axis=0).view(np.float)
        plt.plot(pareto_front[:, 0], pareto_front[:, 1], marker='.')
    plt.title('Pareto Front on partial Wine Dataset')
    plt.xlabel('malic acid')
    plt.ylabel('nonflavanoid phenols')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs = costs3D[:, 0]
    ys = costs3D[:, 1]
    zs = costs3D[:, 2]
    pareto_front = pareto(costs3D)

    ax.scatter(xs, ys, zs)
    ax.scatter(xs[pareto_front], ys[pareto_front], zs[pareto_front], c='orange', marker='X', alpha=1)
    ax.set_xlabel('malic acid')
    ax.set_ylabel('ash')
    ax.set_zlabel('nonflavanoid phenols')
    ax.view_init(10, -15)
    plt.show()
