import numpy as np
import msprime

def stepping_stone(m, length=1, mu=1e-3, n=10, n_rep=1e3):
    '''
    Simulate haploid genotypes under the coalescent model with migration

    Args:
        m: np.array
            d x d array of migration rates
        length: float
            length of chromosome to simulate
        mu: float
            mutation rate
        n: float
            n samples per deme
    Return:
        y: np.array
            (n * d) x p haploid genoytpe matrix
    '''
    # number of demes
    d = m.shape[0]

    # simulate trees
    population_configurations = [msprime.PopulationConfiguration(sample_size=n) for _ in range(d)]
    tree_sequences = msprime.simulate(population_configurations=population_configurations,
                                      migration_matrix=m.tolist(),
                                      length=length,
                                      mutation_rate=mu,
                                      num_replicates=n_rep)

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
    y = np.hstack(genotypes)

    return(y)
