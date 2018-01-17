from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import networkx as nx
import numpy as np

from scipy.linalg import pinvh
from scipy.spatial.distance import pdist, squareform

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm


class Habitat(object):
    """Class for defining, visualzing and computing on a
    habitat which is a directed graph with a set of specified
    edge weights

    Attributes
    ----------
    g : nx directed graph
        directed graph object storing the Habitat
    d : int
        number of nodes in the graph
    m : array
        d x d matrix storing the migration
        rates
    """
    def __init__(self):
        # graph object
        self.g = None

        # number of nodes in the graph
        self.d = None

        # migration matrix storing non-negative edge weights
        self.m = None

        # d x 2 matrix of spatial positions
        self.s = None

    def migration_surface(self):
        """User defined method to define edge weights in the graph
        as this will vary often between different simulations
        """
        raise NotImplementedError("migration_surface is not implemented")

    def get_graph_lapl(self):
        """Computes the graph laplacian which is
        a d x d matrix where L = D - M
        """
        # diagonal matrix storing node degree
        d = np.diag(self.m.sum(axis=1))
        self.l = d - self.m

    def rw_dist(self, q):
        """Computes a random walk distance between nodes
        on the graph defined by the habitat. To compute the
        random walk distance the adjaceny matrix must be symmetric

        Arguments:
            q : array
                d x d graph laplacian matrix L or LL'

        Returns:
            r : array
                d x d array of random walk distances between
                each node
        """
        # invert the graph lapl ... pinvh assumes q is symmetric and psd
        q_inv = pinvh(q)

        # compute the random walk dist
        r = self._cov_to_dist(q_inv)

        return(r)

    def geo_dist(self):
        """Computes geographic distance between nodes
        on the graph defined by the habitat.

        Arguments:
            s : array
                d x 2 array of spatial positions

        Returns:
            r : array
                d x d of geographic distances between each
                node
        """
        r = squareform(pdist(self.s, metric="seuclidean")) / 2
        return(r)

    def coal_dist(self, m):
        """Computes expected genetic distance between nodes
        on the graph defined by the habitat under a coalescent
        stepping stone model for migration with constant population
        sizes

        Arguments:
            m : array
                d x d array of migration rates (edge weights)

        Returns:
            r : array
                d x d of expected genetic distances between each
                node
        """
        pass

    def _cov_to_dist(self, sigma):
        """Converts covariance matrix to distance matrix

        Arguments:
            sigma : np.array
                covariance matrix
        Returns:
            d : np.array
                distance matrix
        """
        n = sigma.shape[0]
        ones = np.ones(n).reshape(n, 1)
        sigma_diag = np.diag(sigma).reshape(n, 1)
        d = ones.dot(sigma_diag.T) + sigma_diag.dot(ones.T) - (2. * sigma)
        return(d)

    def plot_habitat(self, node_size, edge_width_mult, arrows=False):
        """Plot the habitat as weighted directed graph

        Arguments:
            node_size: float
                size of nodes in plot
            edge_width_mult: float
                multiplier of edge weights in plot
        """
        # extract edge weights
        weights = [self.g[i][j]['m'] for i,j in self.g.edges() if self.g[i][j]['m'] != 0.0]

        # extract non-zero edges
        edges = [(i,j) for i,j in self.g.edges() if self.g[i][j]['m'] != 0.0]

        # draw the habitat
        nx.draw(self.g, pos=self.pos_dict, node_size=node_size,
                node_color=(self.s[:,0]**2 + (np.sqrt(self.d) / 2) * self.s[:,1]),
                cmap=cm.viridis, arrows=arrows, edgelist=edges,
                width=edge_width_mult*np.array(weights), edge_color='gray')

    def plot_migration_matrix(self):
        """Plot the migration matrix as a heatmap
        """
        plt.imshow(self.m, cmap=cm.viridis)
        plt.colorbar()

    def plot_precision_matrix(self, q):
        """Plots the precision matrix as a heatmap

        Arguments:
            q : array
                n x n graph laplacian L or LL'
        """
        plt.imshow(q, cmap='seismic', norm=mpl.colors.Normalize(vmin=-np.max(q),
                   vmax=np.max(q)))
        plt.colorbar()


class TriangularLattice(Habitat):
    """Class for a habitat that is a triangular latttice

    Arguments
    ---------
    r: int
        number of rows in the latttice
    c: int
        number of columns in the lattice

    Attributes
    ----------
    g : nx directed graph
        directed graph object storing the Habitat
    d : int
        number of nodes in the graph
    m : array
        d x d matrix storing the migration
        rates
    r : int
        number of rows in the latttice
    c : int
        number of columns in the lattice
    pos_dict : dict
        dictionary of spatial positions
    v : array
        array of node ids
    s : array
        d x 2 array of spatial positions
    """
    def __init__(self, r, c):
        # inherits from Habitat
        super().__init__()

        # number of rows
        self.r = r

        # number of cols
        self.c = c

        # number of nodes
        self.d = self.r * self.c

        # create the graph
        self.g = nx.generators.lattice.triangular_lattice_graph(r - 1, 2 * c - 2, with_positions=True)

        # make node ids ints
        self.g = nx.convert_node_labels_to_integers(self.g)

        # convert to directed graph
        self.g = self.g.to_directed()

        # dictionary of positions
        self.pos_dict = nx.get_node_attributes(self.g, "pos")

        # array of node ids
        self.v = np.array(list(self.g.nodes()))

        # array of spatial positions
        self.s = np.array(list(self.pos_dict.values()))


class Circle(Habitat):
    """Class for a habitat that is a cirlce

    Arguments
    ---------
    d: int
        number of node (demes)

    Attributes
    ----------
    g : nx directed graph
        directed graph object storing the Habitat
    d : int
        number of nodes in the graph
    m : array
        d x d matrix storing the migration
        rates
    pos_dict : dict
        dictionary of spatial positions
    v : array
        array of node ids
    s : array
        d x 2 array of spatial positions
    """
    def __init__(self, d):

        super().__init__()
        
        # number of nodes
        self.d = d

        # create the graph
        self.g = nx.cycle_graph(d)

        # make node ids ints
        self.g = nx.convert_node_labels_to_integers(self.g)

        # convert to directed graph
        self.g = self.g.to_directed()

        # dictionary of positions
        self.pos_dict = nx.circular_layout(self.g)

        # array of node ids
        self.v = np.array(list(self.g.nodes()))

        # array of spatial positions
        self.s = np.array(list(self.pos_dict.values()))
