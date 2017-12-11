import networkx as nx
import numpy as np
from scipy.linalg import pinv

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm


class Habitat(object):
    '''Class for defining the habitat which is a graph'''

    def __init__(self, n, p):
        '''
        Intialize habitat

        Args:
            n: int
                number of row in lattice
            p: int
                number of columns in lattice
        '''
        self.n = n # number of rows
        self.p = p # number of cols
        self.d = self.n * self.p # total number of nodes

        # generate a triangular lattice
        self.g = nx.generators.lattice.triangular_lattice_graph(n - 1, 2 * p - 2, with_positions=True)

        # makde node ids ints
        self.g = nx.convert_node_labels_to_integers(self.g)

        # convert to directed graph
        self.g = self.g.to_directed()

        # dictionary of positions
        self.pos_dict = nx.get_node_attributes(self.g, 'pos')

        # array of node ids
        self.v = np.array(list(self.g.nodes()))

        # array of spatial positions
        self.s = np.array(list(self.pos_dict.values()))

    def mig_rate_func(self):
        '''
        user defined method to define edge weights in the graph
        as the will very often between different simulations
        '''
        raise NotImplementedError('mig_rate_func is not implemented')

    def plot_graph(self, node_size, edge_width_mult, arrows=False):
        '''
        Plot the habitat as weight directed graph

        Args:
            node_size: float
                size of nodes in plot
            edge_width_mult: float
                multiplier of edge weights in plot
        '''
        weights = [self.g[i][j]['m'] for i,j in self.g.edges() if self.g[i][j]['m'] != 0.0]
        edges = [(i,j) for i,j in self.g.edges() if self.g[i][j]['m'] != 0.0]
        nx.draw(self.g, pos=self.pos_dict, node_size=node_size,
                node_color=(self.s[:,0]**2 + (np.sqrt(self.d) / 2) * self.s[:,1]),
                cmap=cm.viridis, arrows=arrows, edgelist=edges,
                width=edge_width_mult*np.array(weights), edge_color='gray')

    def plot_mig_mat(self):
        '''
        Plot a heat map of the migration matrix
        '''
        plt.imshow(self.m, cmap=cm.viridis)
        plt.colorbar()

    def compute_graph_laplacian(self):
        '''
        Computes the graph laplacian using the edge weights
        L = D - M
        '''
        d = np.diag(self.m.sum(axis=1))
        self.l = d - self.m
        self.llt = self.l @ self.l.T

    def compute_distances(self):
        '''
        Computes the resistence distance and
        the random walk distance using the
        graph laplacian
        '''
        # resistence distance
        l_inv = pinv(self.l)
        self.d_res = cov_to_dist(l_inv)

        # random walk disatnce
        llt_inv = pinv(self.llt)
        self.d_rw = cov_to_dist(llt_inv)

    def plot_lapl(self, l):
        '''
        Plots a heatmap of the graph laplacian

        Args:
            l: np.array
                n x n graph laplacian
        '''
        plt.imshow(l, cmap='seismic', norm=mpl.colors.Normalize(vmin=-np.max(l), vmax=np.max(l)))
        plt.colorbar()

def cov_to_dist(sigma):
    '''
    Converts covariance matrix to distance matrix

    Args:
        sigma: np.array
            covariance matrix
    Returns:
        d: np.array
            distance matrix
    '''
    n = sigma.shape[0]
    ones = np.ones(n).reshape(n, 1)
    sigma_diag = np.diag(sigma).reshape(n, 1)
    d = ones.dot(sigma_diag.T) + sigma_diag.dot(ones.T) - (2. * sigma)

    return(d)

