import numpy as np
from sklearn.utils import check_random_state
# https://blog.csdn.net/mmc2015/article/details/51878080
# https://blog.csdn.net/mmc2015/article/details/51835190
# https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html
from sklearn.externals.joblib import Parallel, delayed

# Some reference:
# https://github.com/nicodv/kmodes
# https://github.com/Britefury/binary-kmodes

# A 256 element array that acts as a lookup table that maps an 8-bit number
# `i` to the number of one-bits in the binary representation of `i`
_BIT_COUNT = np.unpackbits(np.arange(256).astype(np.uint8)[:, None],
                           axis=1).sum(axis=1).astype(np.int)


class KPrototypes:

    def __init__(self, n_clusters=8, max_iter=300, n_init=10, gamma=None, tol=1e-4, random_state=0):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.gamma = gamma
        self.tol = tol
        self.random_state = random_state

    def fit(self, Xnum, Xcat):
        rng = check_random_state(self.random_state)

        seeds = rng.randint(np.iinfo(np.int32).max, size=self.n_init)

        results = []

        for i_init in range(self.n_init):
            results.append(self._fit_once(Xnum, Xcat, seeds[i_init]))

        all_inertia = []
        for i in range(self.n_init):
            all_inertia.append(results[i][-1])

        idx = np.argmin(all_inertia)
        best_result = results[idx]
        self.labels_, self.cluster_centers_numerical_, self.cluster_centers_categorical_, self.n_iter_, self.inertia_ \
            = best_result
        return self

    def predict(self, Xnum, Xcat):
        self._check_is_fitted()
        n_samples = Xnum.shape[0]
        assert n_samples == Xcat.shape[0], 'Xnum and Xcat must contain the same number of samples!'
        # Hard-coded batchsize, to prevent heavy memory usage
        batchsize = 5000
        labels = np.zeros(n_samples, dtype=np.int)
        for i_beg in range(0, n_samples, batchsize):
            i_end = min(i_beg + batchsize, n_samples)
            dist = KPrototypes._pairwise_distances(Xnum[i_beg:i_end], Xcat[i_beg:i_end],
                                                   self.cluster_centers_numerical_, self.cluster_centers_categorical_,
                                                   self.gamma)
            labels[i_beg:i_end] = np.argmin(dist, axis=1)
        return labels

    def fit_predict(self, Xnum, Xcat):
        self.fit(Xnum, Xcat)
        return self.labels_

    # Note: This method is not optimized for fit with large n_clusters
    # Refer k_means_.py in sklearn.cluster.KMeans for explanation, distance matrix
    def _fit_once(self, Xnum, Xcat, random_state):

        rng = check_random_state(random_state)

        nnum = Xnum.shape[1]
        ncat = Xcat.shape[1]

        n_samples = Xnum.shape[0]
        assert n_samples == Xcat.shape[0], 'Xnum and Xcat array must have the same number of samples!'

        if self.gamma is None:
            self.gamma = Xnum.std(axis=0).mean() * 1.0 / (ncat * 8)  # 1.0 / (ncat * 8)

        labels = np.zeros((n_samples), dtype=np.int)

        cluster_centers_numerical = np.zeros((self.n_clusters, nnum))
        cluster_centers_categorical = np.zeros((self.n_clusters, ncat), dtype=np.uint8)

        clusterCount = np.zeros(self.n_clusters, dtype=np.int)

        sumOfNumerical = np.zeros((self.n_clusters, nnum))
        frequencyOfCategoricalOne = np.zeros((self.n_clusters, ncat * 8), dtype=np.int)

        ###########################################################################
        # kmeans++ style cluster assignment initialization
        ###########################################################################
        n_local_trials = 2 + int(np.log(self.n_clusters))

        # Choose one sample randomly as cluster center
        center_id = rng.randint(n_samples)
        cluster_centers_numerical[0] = Xnum[center_id]
        cluster_centers_categorical[0] = Xcat[center_id]

        # Compute distances to the first cluster center
        closest_dist = KPrototypes._pairwise_distances(Xnum, Xcat, Xnum[center_id, np.newaxis], Xcat[center_id, np.newaxis], self.gamma).flatten()
        # Total inertia is the sum of distances to the closest cluster center
        current_inertia = closest_dist.sum()

        # Repeat until all (n_clusters) initial centers have been decided
        for c in range(1, self.n_clusters):
            # Draw n_local_trials samples as new candidate cluster centers, with probability proportional to closest_dist
            rand_vals = rng.random_sample(n_local_trials) * current_inertia
            candidate_ids = np.searchsorted(np.cumsum(closest_dist),
                                            rand_vals)
            # Compute distances to these candidate centers
            dist_to_candidates = KPrototypes._pairwise_distances(Xnum[candidate_ids], Xcat[candidate_ids], Xnum, Xcat,
                                                                 self.gamma)

            # The candidate with minimum total potential is best
            best_candidate = None
            best_pot = None
            best_dist = None

            for i_trial in range(n_local_trials):
                # dist when add the candidate as a new center
                new_closest_dist = np.minimum(closest_dist, dist_to_candidates[i_trial])
                new_pot = new_closest_dist.sum()

                if (best_candidate is None) or (new_pot < best_pot):
                    best_candidate = candidate_ids[i_trial]
                    best_pot = new_pot
                    best_dist = new_closest_dist

            cluster_centers_numerical[c] = Xnum[best_candidate]
            cluster_centers_categorical[c] = Xcat[best_candidate]

            current_inertia = best_pot
            closest_dist = best_dist

        ###########################################################################
        # Iteration
        ###########################################################################
        old_inertia = current_inertia
        i_iter = 0
        converged = False
        while (i_iter <= self.max_iter) and (not converged):
            i_iter += 1

            moves = 0

            for i_sample in range(n_samples):
                # nn
                to_cluster, dist = KPrototypes._nearest_cluster(cluster_centers_numerical, cluster_centers_categorical,
                                                                Xnum[i_sample], Xcat[i_sample], self.gamma)
                current_inertia += dist
                if labels[i_sample] == to_cluster:
                    continue

                moves += 1
                from_cluster = labels[i_sample]
                clusterCount[to_cluster] += 1
                clusterCount[from_cluster] -= 1

                # Update numerical
                sumOfNumerical[to_cluster] += Xnum[i_sample]
                sumOfNumerical[from_cluster] -= Xnum[i_sample]
                cluster_centers_numerical[to_cluster]
                cluster_centers_numerical[from_cluster]

                # Update categorical
                frequencyOfCategoricalOne[to_cluster] += np.unpackbits(Xcat[i_sample])
                frequencyOfCategoricalOne[from_cluster] -= np.unpackbits(Xcat[i_sample])
                cluster_centers_categorical[to_cluster] = np.packbits((frequencyOfCategoricalOne[to_cluster] >=
                                                                       clusterCount[to_cluster] -
                                                                       frequencyOfCategoricalOne[to_cluster]))
                cluster_centers_categorical[from_cluster] = np.packbits((frequencyOfCategoricalOne[from_cluster] >=
                                                                         clusterCount[from_cluster] -
                                                                         frequencyOfCategoricalOne[from_cluster]))

            converged = (moves == 0) or (old_inertia - current_inertia < self.tol * old_inertia)
            old_inertia = current_inertia
            if converged:
                break

        return labels, cluster_centers_numerical, cluster_centers_categorical, i_iter, current_inertia

    # todo: 把这个改成batch式的
    @staticmethod
    def _nearest_cluster(cluster_centers_numerical, cluster_centers_categorical, Xnum, Xcat, gamma):
        dists = np.sum((cluster_centers_numerical - Xnum) ** 2, axis=1) + gamma * _BIT_COUNT[
            np.bitwise_xor(cluster_centers_categorical, Xcat)].sum(axis=1)
        cluster = np.argmin(dists)
        return cluster, dists[cluster]

    @staticmethod
    def _pairwise_distances(Xnum, Xcat, Ynum, Ycat, gamma):
        nx = Xnum.shape[0]
        ny = Ynum.shape[0]

        nnum = Xnum.shape[1]
        ncat = Xcat.shape[1]

        #dist = pairwise_distances(Xnum, Ynum, metric='euclidean', n_jobs=-1)

        #pairwise_distanddces(Xcat, Ycat, metric='hamming', n_jobs=-1)

        dist = np.square(Xnum[0:nx, np.newaxis, :] - Ynum[np.newaxis, 0:ny, :]).sum(axis=-1)
        dist += gamma * _BIT_COUNT[np.bitwise_xor(Xcat[0:nx, np.newaxis, :], Ycat[np.newaxis, 0:ny, :])].sum(axis=-1)
        return dist


    @staticmethod
    # Arithmetic mean for numerical data, mode for categorical data
    def _mean_mode(Xnum, Xcat, batchsize=None):
        n_samples = Xnum.shape[0]

        # If batchsize is not assigned, perform categorical mode calculation of all samples at one time
        if batchsize is None:
            batchsize = n_samples

        Mnum = Xnum.mean(axis=0)

        zeros = None
        ones = None
        for i in range(0, n_samples, batchsize):
            Xcat_batch = Xcat[i:i + batchsize]
            Xcat_batch_unpacked = np.unpackbits(Xcat_batch, axis=1)
            o = Xcat_batch_unpacked.sum(axis=0)
            z = batchsize - o
            zeros = (zeros + z) if zeros is not None else z
            ones = (ones + o) if ones is not None else o

        Mcat = np.packbits(ones >= zeros)

        return Mnum, Mcat

    def _check_is_fitted(self):
        assert hasattr(self, 'cluster_centers_numerical_') and hasattr(self,
                                                                       'cluster_centers_categorical_'), 'Model is not fitted yet!'
