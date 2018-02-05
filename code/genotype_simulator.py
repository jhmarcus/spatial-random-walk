from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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


class GenotypeSimulator(object):
    """Class for simulating genotypes under the coalescent
    given a habitat, a directed graph which individuals migrate
    over

    Arguments
    ---------
    hab : Habitat
        habitat object
    sim_path: str
        path to simulation pkl file
    chrom_length: float
        length of chrom to simulate
    mu: float
        mutation rate
    n_samp: int
        n haploid samples per deme
    n_rep: int
        number of indepdent regions to simulate from
    eps: float
        min derived allele frequency for filtering out rare variants

    Attributes
    ----------
    hab : Habitat
        habitat object
    chrom_length: float
        length of chrom to simulate
    mu: float
        mutation rate
    n_samp: int
        n haploid samples per deme
    n_rep: int
        number of indepdent regions to simulate from
    eps: float
        min derived allele frequency for filtering out rare variants
    y : array
        n x p genotype matrix
    tree_sequences :
        geneologies object
    n : int
        number of individuals
    p : int
        number of snps
    """
    def __init__(self, hab, sim_path, chrom_length=1, mu=1e-3, n_e=1,
                 n_samp=10, n_rep=1e4, eps=.05):

        # habitat object
        self.hab = hab

        # choromosome length
        self.chrom_length = chrom_length

        # mutation rate
        self.mu = mu

        # effective sizes
        self.n_e = n_e

        # number of haploids per deme
        self.n_samp = n_samp

        # number of indepdent chunks to simulate
        self.n_rep = n_rep

        # min derived allele frequency to filter out
        self.eps = eps

        # if the simulation was already performed extract genotypes
        if os.path.exists(sim_path):
            with open(sim_path, 'rb') as geno:
                self.y = pkl.load(geno)
        # otherwise run the simulation
        else:
            # simulate geneologies from the defined model
            self._simulate_trees()
            self._simulate_genotypes()
            with open(sim_path, 'wb') as geno:
                pkl.dump(self.y, geno)

        # number of snps
        self.n, self.p = self.y.shape

        # node ids for each individual
        self.v = np.repeat(self.hab.v, int(self.n / self.hab.d)).T

        # spatial positions for each individual
        self.s = np.vstack([np.repeat(self.hab.s[:,0], int(self.n / self.hab.d)),
                            np.repeat(self.hab.s[:,1], int(self.n / self.hab.d))]).T

    def _simulate_trees(self):
        """Simulate trees under the coalescent migration model
        defined in the habitat with constant population
        sizes
        """
        # simulate trees
        population_configurations = [msprime.PopulationConfiguration(sample_size=self.n_samp) for _ in range(self.hab.d)]
        self.tree_sequences = msprime.simulate(population_configurations=population_configurations,
                                               migration_matrix=self.hab.m.tolist(),
                                               length=self.chrom_length,
                                               mutation_rate=self.mu,
                                               num_replicates=self.n_rep,
                                               Ne=self.n_e)

    def _simulate_genotypes(self):
        """Extract trees and simulate mutations in each
        independent region to obtain a genotype matrix
        """
        # extract mutations
        genotypes = []

        # loop through each region
        for i,tree_sequence in enumerate(self.tree_sequences):

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
        print("n={},p={}".format(self.y.shape[0], self.y.shape[1]))

    def filter_rare_var(self):
        """Filter out rare variants
        """
        daf = np.sum(self.y, axis=0) / self.n
        idx = np.where((daf >= self.eps) & (daf <= (1. - self.eps)))[0]
        self.y = self.y[:,idx]
        self.n, self.p = self.y.shape

    def geno_dist(self):
        """Compute the observed genetic distance between
        individuals

        Returns
        -------
        d_gen : array
            n x n array of observed genetic distances for each
            pair
        """
        # mean frequencies for each snp
        mu = np.mean(self.y, axis=0, keepdims=True)
        d_gen = squareform(pdist((self.y - mu), metric='seuclidean')) / self.p

        return(d_gen)

    def pca(self):
        """Run principal components analysis
        on the genotype matrix
        """
        mu = np.mean(self.y, axis=0)
        std = np.std(self.y, axis=0)
        z = (self.y - mu) / std
        pca = PCA(n_components=50)
        pca.fit(z.T)
        self.pcs = pca.components_.T
        self.pves = pca.explained_variance_ratio_

    def plot_sfs(self):
        """Plot the observed site frequency spectrum and neutral expectation
        """
        dac = np.sum(self.y, axis=0)
        x = np.arange(1, self.n) / self.n
        sfs = np.histogram(dac, bins=np.arange(1, self.n + 1))[0]
        plt.semilogy(x, sfs / sfs[0], '.')
        plt.semilogy(x, 1 / (x * self.n), '--')
        plt.xlabel('Derived Allele Frequency')
        plt.ylabel('log(Count)')

    def plot_pca(self, pcs, pves, figsize=(12, 6)):
        """Plot PC1 vs PC2 and scree plot

        Arguments:
            pcs : array
                pcs output from pca
            pves : array
                proportion of variance explained for each pc
        """
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=False, figsize=figsize)

        # figure 1
        ax1.scatter(pcs[:,0], pcs[:,1], c=self.s[:,0]**2 + (np.sqrt(self.hab.d) / 2) * self.s[:,1], cmap=cm.viridis)
        ax1.set_xlabel('PC1 ({})'.format(np.round(pves[0], 4)))
        ax1.set_ylabel('PC2 ({})'.format(np.round(pves[1], 4)))

        # figure 2
        ax2.scatter(np.arange(pves.shape[0]), pves)
        ax2.set_xlabel('PC')
        ax2.set_ylabel('PVE')

    def plot_dist(self, d_x_tril, d_y_tril, lab_x, lab_y):
        """
        """
        fit = np.polyfit(d_x_tril, d_y_tril, 1)
        plt.scatter(d_x_tril, d_y_tril, marker='.', alpha=.5)
        plt.plot(d_x_tril, fit[0] * d_x_tril + fit[1], c='orange')
        plt.xlabel(lab_x)
        plt.ylabel(lab_y)

    def node_to_obs_mat(self, x, n, v):
        """Converts node level array to data level array

        Arguments:
            x : array
                array at the level of nodes
            n : int
                number of observations
            v : array
                array carraying the node ids for each
                observation
        Returns:
            y : array
                array at the level of observations repeated
                from the node level array
        """
        y = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                y[i,j] = x[v[i], v[j]]

        return(y)
