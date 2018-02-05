from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import networkx as nx
import numpy as np

from scipy.linalg import pinvh
from scipy.sparse.linalg import cg
from scipy.sparse import csr_matrix
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
        a d x d matrix where L = I - M as M is markov
        matrix its rows sum to 1
        """
        # adding diagonal to migration matrix
        m = np.zeros((self.d, self.d))
        diag =  1. - np.sum(self.m, axis=1)
        diag_idx = np.diag_indices(self.d)

        m = np.array(self.m.tolist())
        m[diag_idx] = diag

        self.l = np.eye(self.d) - m

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

    def coal_dist(self, tol=1e-8):
        """Computes expected genetic distance between nodes
        on the graph defined by the habitat under a coalescent
        stepping stone model for migration with constant population
        sizes

        Arguments:
            tol : float
                tolerence for solving linear system using conjugate gradient
        Returns:
            t : array
                d x d of expected genetic distances between each
                node
        """
        # upper tri indicies including diagonal
        triu_idx = np.triu_indices(self.d, 0)

        # number of within deme equations and between deme equations
        n_wb = triu_idx[0].shape[0]

        # d x d matrix storing indicies of each pair
        h = np.zeros((self.d, self.d), dtype=np.int64)
        k = 0
        for i in range(self.d):
            for j in range(i, self.d):
                h[i, j] = k
                h[j, i] = k
                k += 1

        # coefficents of coal time equation
        A = np.zeros((n_wb, n_wb))

        # solution to coal time equation
        b = np.ones(n_wb)

        # loop of all unique pairs of demes
        for i in range(n_wb):

            # deme pair for each row
            alpha, beta = (triu_idx[0][i], triu_idx[1][i])

            if alpha == beta:
                c = h[alpha, beta]
                A[i, c] += 1.  # add coalescent rate

            # loop over neighbors of deme alpha
            for gamma in range(self.d):
                c = h[beta, gamma]
                A[i, c] += self.l[alpha, gamma]

            # loop over the neighbors of deme beta
            for gamma in range(self.d):
                c = h[alpha, gamma]
                A[i, c] += self.l[beta, gamma]

        #t = np.empty((self.d, self.d))
        #t[triu_idx] = np.linalg.solve(A, b)
        #t = t + t.T - np.diag(np.diag(t))

        A_ = csr_matrix(A)

        t_ = cg(A_, b, tol=tol)
        t = np.empty((self.d, self.d))
        t[triu_idx] = t_[0]
        t = t + t.T - np.diag(np.diag(t))

        return(t)

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

class SquareLattice(Habitat):
    """Class for a habitat that is a square latttice

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
        self.g = nx.grid_2d_graph(self.r, self.c)

        # dictionary of positions
        self.pos_dict = {}
        for i,node in enumerate(self.g.nodes):
            self.g.nodes[node]["pos"] = node
            self.pos_dict[i] = node

        #nx.set_node_attributes(self.g, "pos", self.pos_dict)

        # make node ids ints
        self.g = nx.convert_node_labels_to_integers(self.g)

        # convert to directed graph
        self.g = self.g.to_directed()

        # array of node ids
        self.v = np.array(list(self.g.nodes()))

        # array of spatial positions
        self.s = np.array(list(self.pos_dict.values()))


class Line(Habitat):
    """Class for a habitat that is a square latttice

    Arguments
    ---------
    d: int
        number of nodes in the lattice

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

        # inherits from Habitat
        super().__init__()

        # number of nodes
        self.d = d

        # create the graph
        self.g = nx.grid_graph([self.d])

        # dictionary of positions
        self.pos_dict = {}
        for i,node in enumerate(self.g.nodes):
            self.g.nodes[node]["pos"] = (node, 0.)
            self.pos_dict[i] = (node, 0.)

        #nx.set_node_attributes(self.g, "pos", self.pos_dict)

        # make node ids ints
        self.g = nx.convert_node_labels_to_integers(self.g)

        # convert to directed graph
        self.g = self.g.to_directed()

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
