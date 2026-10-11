import numpy as np
from misc.disc_dist_divergences import minimum_hamming_distance
from misc.disc_sample import sample_cat


def test_hamming_dist():

    for _i in range(100):
        # create some distribution
        num_modes = 10
        dist = np.random.randint(0, 1000, num_modes)

        # now sample a sequence from the distribution
        num_elements = 1000
        samples = sample_cat(dist, num_elements)
        o_samples = np.copy(samples)

        # now create random numer
        h_dist = np.random.randint(0, 100)
        p = np.random.permutation(1000)

        # change these elements
        for k in range(h_dist):
            r = p[k]
            s = samples[r]
            old_dist = dist[s]
            dist[s] = 0
            samples[r] = sample_cat(dist)
            dist[s] = old_dist

        res, _ = minimum_hamming_distance(samples, o_samples)
        assert res == h_dist


def test_misc():
    test_hamming_dist()


test_misc()
