import numpy as np
import scipy
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from pyriemann.utils.distance import distance_riemann


# class RSA():

#     def __init__(self,rdm1,rdm2,similarity_measure) -> None:
#         self.rdm1 = rdm1
#         self.rdm2 = rdm2

#         self.similarity_measure = similarity_measure

#     def __call__(self):

#         if self.similarity_measure == 'pearsonr':
#             self.rsa = pearsonr(self.rdm1,self.rdm2)
#         elif self.similarity_measure == 'spearmanr':
#             self.rsa = spearmanr(self.rdm1,self.rdm2)

#         return self.rsa


# class RiemannRSA(RSA):
#     def __init__(self, rdm1, rdm2, similarity_measure) -> None:
#         super().__init__(rdm1, rdm2, similarity_measure)

#         assert self.similarity_measure = 'riemann_distance'

#     def __call__(self):
#         import pyriemann #lazy import


def shuffle_matrix(x: np.array) -> np.array:
    """
    Simple utility to shuffle a (M,M) matrix across rows and columns,
    i.e., the order in the matrix is changed but the information (in 
    a particular row/col) is not destroyed
    """

    shuffled_x = x.copy()
    sort_idx = np.random.permutation(shuffled_x.shape[0])

    shuffled_x = shuffled_x[sort_idx, :]
    shuffled_x = shuffled_x[:, sort_idx]

    assert np.all(shuffled_x == shuffled_x.T)
    return shuffled_x


def get_rdm(reps: list, metric: str = "correlation"):
    """
    Get an RDM with the features
    Args:
        distance : argument for the pdist function of scipy to calculate the pdist
    """
    # assert reps is not None
    # assert reps != []
    return pdist(reps, metric=metric)


def get_rsm(reps: list, distance: str = "correlation"):
    """
    Get an RSM with the features
    Args:
        distance : argument for the pdist function of scipy to calculate the pdist
    """
    rdm = get_rdm(reps, distance=distance)
    rsm = 1.0 - squareform(rdm)
    rsm = squareform(rsm)
    assert len(rsm.shape) == 1
    return rsm


def bias_corrected_driem(rsm_a:np.array, rsm_b:np.array) -> dict:

    # matlab code from the authors
    # ------------------
    # dist = distance_riemann(A, B);

    # % permutation
    # if do_perm
    #     perm_dists = nan(n_rept, 1);
    #     for perm_i = 1 : n_rept
    #         rnd_perm = randperm(L);
    #         perm_dists(perm_i) = distance_riemann(A, B(rnd_perm, rnd_perm));
    #     end

    #     % tstat & p-value calculation
    #     p = 0; %ranksum(perm_dists, dist, 'tail', 'right', 'method', 'exact');

    #     mu = mean(perm_dists);
    #     sigma = std(perm_dists);
    #     z = (mu - dist)/(sigma);
    #     bias_corrected_dist = mu - dist;
    # else
    #     p = nan;
    #     z = nan;
    #     bias_corrected_dist = nan;
    #     perm_dists = nan;
    # ------------------

    dist = distance_riemann(rsm_a, rsm_b)
    perm_dists = [
        distance_riemann(rsm_a, shuffle_matrix(rsm_b)) for _ in range(10000)
    ]  # acts as our 'baseline for comparisons'
    z = np.std(np.array(perm_dists))

    bias_corrected_dist = {
        # FIXME: shouldn't this be (dist - mu)/z?
        "bias_corrected_dist": (np.mean(perm_dists) - dist) / z,
        "mu": np.mean(perm_dists),
        "z": z,
    }

    return bias_corrected_dist


def get_rsa(rdm1, rdm2, similarity_measure="pearsonr"):

    assert len(rdm1.shape) == 1  # that the rdms are not in squareform
    assert rdm1.shape == rdm2.shape

    if similarity_measure == "pearsonr":
        return pearsonr(rdm1, rdm2)
    if similarity_measure == "spearmanr":
        return spearmanr(rdm1, rdm2)
    if similarity_measure == "euclidean":
        return np.mean(
            (rdm1 - rdm2) ** 2
        )  # you can also scale the mean with the dimensionality of the matrices (by sqrt(d) for example)
    if similarity_measure == "riemann":
        print("Since the args are already rdms, converting them into rsms")
        rsm1 = 1.0 - squareform(rdm1)
        rsm2 = 1.0 - squareform(rdm2)

        assert len(rsm1.shape) == 2  # that the rsms are PSDs
        assert len(rsm2.shape) == 2  # that the rsms are PSDs

        if similarity_measure == "bias_corrected":
            return bias_corrected_driem(rsm1, rsm2)
        else:
            return distance_riemann(rsm1, rsm2)
