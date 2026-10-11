import numpy as np

from misc.disc_dist_divergences import symm_kl_divergence


def sample_cat(measure, shape=None, use_logits=False):
    """A simple categorical sampler, uses un-normalized measure.

    :param measure: A vector [L] of a measure or a matrix [MxL], where each row is one measure
    :param shape: A tuple of which shape should be sampled for each passed measure.
    :param use_logits: True if the values are already in log space.
    :return: A matrix [MxS[0]x...xS[D-1]] of type int64 which holds the sampled values v with 0 <= v < L.
    """
    # some properties
    exp_measure = False

    # expand if necessary
    if np.ndim(measure) == 1:
        exp_measure = True
        measure = np.expand_dims(measure, 0)

    # for the shape part
    if shape is None:
        shape = np.ones([0], np.int64)
    elif np.ndim(shape) == 0:
        shape = np.expand_dims(shape, 0)

    # assertion for the categorical sampler
    assert np.ndim(measure) == 2
    assert np.ndim(shape) == 1

    # get some information
    num_measures, max_state = measure.shape

    # map to log space using smoothed measure
    smoothed_measure = (measure + 0.0001) / (1 + 0.0001)
    if not use_logits:
        alpha = np.log(smoothed_measure)
    else:
        alpha = smoothed_measure

    # concat the sizes and generate gumbel samples
    # by inverse transformation
    r_size = np.concatenate(([num_measures], shape, [max_state]))
    u = np.random.uniform(size=r_size)
    g = -np.log(-np.log(u))

    # get num dimensions and finalize
    num_dim = np.ndim(r_size) - 2
    alpha = alpha.reshape([num_measures] + [1] * num_dim + [max_state])
    res = np.argmax(g + alpha, axis=-1)
    return res if not exp_measure else res[0]


def test_sample_cat():

    # create measure
    measure = np.random.uniform(0, 1.0, 100)
    nmeasure = measure / np.sum(measure)

    samples = sample_cat(measure, 300000, use_logits=False)
    samples = np.bincount(samples)
    estmeasure = samples / np.sum(samples)
    print(symm_kl_divergence(nmeasure, estmeasure))


if __name__ == "__main__":
    test_sample_cat()
