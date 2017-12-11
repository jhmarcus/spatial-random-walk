import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

import msprime
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform

import pickle as pkl
import os


class SteppingStone(object):
    '''
    Class for simulating genotypes under the stepping stone model
    and computing and visualizing a variety of summeries of the data
    particularly related to predictors of genetic distance
    '''

    def __init__(self, h, sim_path, length=1, mu=1e-3, n_samp=10, n_rep=1e3, eps=.05):
        '''
        Intialize stepping stone simulation

        Args:
            h: Habitat
                habitat object
            sim_path: str
                path to simulation pkl file
            length: float
                length of chrom to simulate
            mu: float
                mutation rate
            n_samp: int
                n haploid samples per deme
            n_rep: int
                number of indepdent regions to simulate from
            eps: float
                min derived allele frequency for filtering out rare variants
        '''
        self.h = h
        self.length = length
        self.mu = mu
        self.n_samp = n_samp
        self.n_rep = n_rep
        self.eps = eps

        if os.path.exists(sim_path):
            with open(sim_path, 'rb') as geno:
                self.y = pkl.load(geno)
        else:
            self._simulate_genotypes()
            with open(sim_path, 'wb') as geno:
                pkl.dump(self.y, geno)

        self.n, self.p_tot = self.y.shape

        self.v = np.repeat(self.h.v, int(self.n / self.h.d)).T
        self.s = np.vstack([np.repeat(self.h.s[:,0], int(self.n / self.h.d)),
                            np.repeat(self.h.s[:,1], int(self.n / self.h.d))]).T

    def _simulate_trees(self):
        '''
        Simulate trees from the coalescent using msprime
        '''
        # simulate trees
        population_configurations = [msprime.PopulationConfiguration(sample_size=self.h.n) for _ in range(self.h.d)]
        tree_sequences = msprime.simulate(population_configurations=population_configurations,
                                          migration_matrix=self.h.m.tolist(),
                                          length=self.length,
                                          mutation_rate=self.mu,
                                          num_replicates=self.n_rep)

        return(tree_sequences)

    def _simulate_genotypes(self):
        '''
        Extract trees and simulate mutations in each
        independent region
        '''
        # simulate trees
        tree_sequence = _simulate_trees()

        # extract mutations
        genotypes = []

        # loop through each region
        for i,tree_sequence in enumerate(tree_sequences):

            if i % 250 == 0:
                print('extracting tree {}'.format(i))

            shape = tree_sequence.get_num_mutations(), tree_sequence.get_sample_size()
            g = np.empty(shape, dtype="u1")

            # loop through each tree
            for variant in tree_sequence.variants():
                g[variant.index] = variant.genotypes

            genotypes.append(g.T)

        # (n*d) x p genotype matrix
        self.y = np.hstack(genotypes)

    def plot_sfs(self):
        '''
        Plot the observed site frequency spectrum and neutral expectation
        '''
        dac = np.sum(self.y, axis=0)
        x = np.arange(1, self.n) / self.n
        sfs = np.histogram(dac, bins=np.arange(1, self.n + 1))[0]
        plt.semilogy(x, sfs / sfs[0], '.')
        plt.semilogy(x, 1 / (x * self.n), '--')
        plt.xlabel('Derived Allele Frequency')
        plt.ylabel('log(Count)')

    def filter_rare_var(self):
        '''
        Filter out rare variants
        '''
        daf = np.sum(self.y, axis=0) / self.n
        idx = np.where((daf >= self.eps) & (daf <= (1. - self.eps)))[0]
        self.y = self.y[:,idx]
        self.n, self.p = self.y.shape

    def plot_pca(self, figsize=(12, 6)):
        '''
        Run pca on normalized genotype data
        '''
        mu = np.mean(self.y, axis=0)
        std = np.std(self.y, axis=0)
        z = (self.y - mu) / std
        pca = PCA(n_components=50)
        pca.fit(z.T)
        pcs = pca.components_.T
        pves = pca.explained_variance_ratio_

        f, (ax1, ax2) = plt.subplots(1, 2, sharey=False, figsize=figsize)

        # figure 1
        ax1.scatter(pcs[:,0], pcs[:,1], c=self.s[:,0]**2 + (np.sqrt(self.h.d) / 2) * self.s[:,1], cmap=cm.viridis)
        ax1.set_xlabel('PC1 ({})'.format(np.round(pves[0], 4)))
        ax1.set_ylabel('PC2 ({})'.format(np.round(pves[1], 4)))

        # figure 2
        ax2.scatter(np.arange(pves.shape[0]), pves)
        ax2.set_xlabel('PC')
        ax2.set_ylabel('PVE')

    def compute_distances(self):
        '''
        Compute relevent distances
        '''
        mu = np.mean(self.y, axis=0)
        self.d_gen = squareform(pdist((self.y - mu), metric='seuclidean')) / self.p
        self.d_geo = squareform(pdist(self.s, metric='seuclidean')) / 2
        self.d_res = node_to_obs_mat(self.h.d_res, self.n, self.v)
        self.d_rw = node_to_obs_mat(self.h.d_rw, self.n, self.v)

    def plot_distances(self, figsize=(14,4)):
        '''
        Plot subplots of x distance vs genetic distance

        Args:
            figsize: tuple
                total figure size (width, height)
        '''
        tril_idx = np.tril_indices(self.n, -1)
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=figsize)

        # figure 1
        print('geo r2 = {}'.format(np.corrcoef(self.d_geo[tril_idx], self.d_gen[tril_idx])[0, 1]))
        fit = np.polyfit(self.d_geo[tril_idx], self.d_gen[tril_idx], 1)
        ax1.scatter(self.d_geo[tril_idx], self.d_gen[tril_idx], marker='.', alpha=.5)
        ax1.plot(self.d_geo[tril_idx], fit[0] * self.d_geo[tril_idx] + fit[1], c='orange')
        ax1.set_xlabel('Geographic Distance')
        ax1.set_ylabel('Genetic Distance')

        # figure 2
        print('res r2 = {}'.format(np.corrcoef(self.d_res[tril_idx], self.d_gen[tril_idx])[0, 1]))
        fit = np.polyfit(self.d_res[tril_idx], self.d_gen[tril_idx], 1)
        ax2.plot(self.d_res[tril_idx], fit[0] * self.d_res[tril_idx] + fit[1], c='orange')
        ax2.scatter(self.d_res[tril_idx], self.d_gen[tril_idx], alpha=.5, marker='.')
        ax2.set_xlabel('Resistence Distance')

        # figure 3
        print('rw r2 = {}'.format(np.corrcoef(self.d_rw[tril_idx], self.d_gen[tril_idx])[0, 1]))
        fit = np.polyfit(self.d_rw[tril_idx], self.d_gen[tril_idx], 1)
        ax3.plot(self.d_rw[tril_idx], fit[0] * self.d_rw[tril_idx] + fit[1], c='orange')
        ax3.scatter(self.d_rw[tril_idx], self.d_gen[tril_idx], alpha=.5, marker='.')
        ax3.set_xlabel('Random Walk Distance')

    def plot_dist_hist(self, bins=20, figsize=(14,4)):
        '''
        Plot subplots of x distance vs genetic distance

        Args:
            figsize: tuple
                total figure size (width, height)
        '''
        tril_idx = np.tril_indices(self.n, -1)
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=figsize)

        # figure 1
        ax1.hist(self.d_geo[tril_idx], bins=bins)
        ax1.set_xlabel('Geographic Distance')
        ax1.set_ylabel('Count')

        # figure 2
        ax2.hist(self.d_res[tril_idx], bins=bins)
        ax2.set_xlabel('Resistence Distance')

        # figure 3
        ax3.hist(self.d_rw[tril_idx], bins=bins)
        ax3.set_xlabel('Random Walk Distance')

def node_to_obs_mat(x, n, v):
    '''
    Converts node level array to data level array

    Args:
        x: np.array
            array at the level of nodes
        n: int
            number of observations
        v: np.array
            array carraying the node ids for each
            observation
    Returns:
        y: np.array
            array at the level of observations repeated
            from the node level array
    '''
    y = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            y[i,j] = x[v[i], v[j]]

    return(y)

