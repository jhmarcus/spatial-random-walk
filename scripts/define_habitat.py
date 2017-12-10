import math
import numpy as np
import networkx as nx

def gen_lattice(n, p):
    '''
    Generate a n x p triangular regular lattice

    Args:
        n: int
            number of rows in the lattice
        p: int
            number of columns in the lattice
    Returns:
        g: networkx.graph
            regular lattice
    '''
    # convert input for networkx genarator
    n = n - 1
    p = 2 * p - 2

    # define lattice
    g = nx.generators.lattice.triangular_lattice_graph(n, p, with_positions=True)
    g = nx.convert_node_labels_to_integers(g)

    # dictionary of positions
    pos_dict = nx.get_node_attributes(g, 'pos')

    # array of node ids
    v = np.array(list(g.nodes()))

    # array of spatial positions
    s = np.array(list(pos_dict.values()))

    res_dict = {'g': g, 's': s,
                'v': v, 'pos_dict': pos_dict}

    return(res_dict)

def quadratic_barrier_weights(g, s, m_min, m_max):
    '''
    Args:
        g: nx.graph
            regular lattice
        s: np.array
            d x 2 array of spatial positions
    Returns:
        g: nx.graph
            regular lattice with assigned weights
    '''
    s0_max = np.max(s[:,0])
    s0_med = np.median(s[:,0]) + .25
    for i,j in g.edges():
        mu = np.mean([s[i,0], s[j,0]])
        m = (s0_max / s0_med ** 2) * (mu - s0_med) ** 2 + m_min
        g[i][j]['m'] = min(m, m_max)

    return(g)

def asymetric_uniform_migration(g, m_unf):
    '''
    Args:
        g: nx.graph
            regular lattice
        m_unf: float
            uniform migration rate
    Returns:
        g: nx.graph
            regular lattice with assigned weights
    '''
    for i,j in g.edges():
        print(i,j)
        if i < j:
            g[i][j]['m'] = m_unf
        else:
            g[i][j]['m'] = 0.0

    return(g)






