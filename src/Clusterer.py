import numpy as np
from kprototypes import KPrototypes
from sklearn.utils import check_random_state
from time import time
import os
from utilities import WriteImg, Brief

_patchSize = (33, 17, 5)
#_nbytes = (3, 5, 2)
_nbytes = (6, 10, 4)
_sigma = (4.0, 2.0, 0.0)

class Clusterer:
    def __init__(self, img_ambient, size_x, size_y, n_clusters=1024, random_state=0):


        self.n_clusters = n_clusters
        self.img_ambient = img_ambient
        self.size_x = size_x
        self.size_y = size_y

        self.cluster_centers_numerical = None
        self.cluster_centers_categorical = None
        self.labels = None

        self.rng = check_random_state(random_state)

        # self.len_numerical = 3
        # self.len_categorical = 0

    @classmethod
    def FromNpyFile(cls, img_ambient, npy_path, size_x, size_y):
        clusterer = cls(img_ambient, size_x, size_y)
        npy_path = npy_path.rstrip(r'\/')
        npy_file_name = npy_path + '/labels.npy'
        assert clusterer.LoadClusterInfo(npy_path), 'Loading clustering from \'{}\' failed!'.format(npy_file_name)
        return clusterer




    # Fit a model with given number of samples
    def Cluster(self, n_samples=None):

        features_numerical = np.copy(self.img_ambient).reshape(self.size_y*self.size_x, -1)
        t0 = time()
        print('Calculating BRIEF features...')
        features_categorical = Brief(self.img_ambient, _patchSize, _nbytes, _sigma).reshape(self.size_y*self.size_x, -1)
        print('Done in {:.3f}s'.format(time()-t0))


        # Draw n samples from the image
        if n_samples is None:
            samples_numerical = features_numerical
            samples_categorical = features_categorical
            print("Fitting samples of the whole ambient image into {:d} clusters".format(self.n_clusters))

        else:
            n_samples = min(n_samples, self.size_y * self.size_x)
            idx_samples = self.rng.randint(self.size_y * self.size_x, size=(n_samples,))
            samples_numerical = features_numerical[idx_samples]
            samples_categorical = features_categorical[idx_samples]
            print("Fitting {:d} samples of ambient image into {:d} clusters".format(n_samples, self.n_clusters))

        # Fit a model on these samples
        t0 = time()
        kprototypes = KPrototypes(n_clusters=self.n_clusters, random_state=self.rng.randint(np.iinfo(np.int32).max)).fit(samples_numerical, samples_categorical)
        t1 = time()
        print('Done in {:.3f}s.'.format(t1-t0))


        # And predict to which cluster each pixel belongs
        print('Predicting to which cluster each sample of the image belongs')
        self.cluster_centers_numerical = kprototypes.cluster_centers_numerical_
        self.cluster_centers_categorical = kprototypes.cluster_centers_categorical_
        self.labels = kprototypes.predict(features_numerical, features_categorical)
        t2 = time()
        print('Done in {:.3f}s.'.format(t2-t1))

        return self



    # Return samples' indices that belong to cluster n, in (iy, ix) form
    def Indices(self, n):
        idx = np.where(self.labels == n)
        iy = idx[0] // self.size_x
        ix = idx[0] % self.size_x
        return iy, ix

    def LoadClusterInfo(self, path):
        path = path.rstrip(r'\/')
        npy_file = path + '/labels.npy'
        if os.path.isfile(npy_file):

            # Load cluster assignment data from .npy file
            self.labels = np.load(npy_file)
            features_map_numerical = np.copy(self.img_ambient)
            t0 = time()
            print('Calculating BRIEF features...')
            features_map_categorical = Brief(self.img_ambient, _patchSize, _nbytes, _sigma)
            print('Done in {:.3f}s'.format(time() - t0))
            # Recalculate cluster centers
            t0 = time()
            print('Found labels.npy. Recalculating cluster centers...')
            self.cluster_centers_numerical = np.zeros((self.n_clusters, 3))
            self.cluster_centers_categorical = np.zeros((self.n_clusters, sum(_nbytes)), dtype=np.uint8)

            for n in range(self.n_clusters):
                iy, ix = self.Indices(n)
                Xnum = features_map_numerical[(iy, ix)]
                Xcat = features_map_categorical[(iy, ix)]
                self.cluster_centers_numerical[n], self.cluster_centers_categorical[n] = KPrototypes._mean_mode(Xnum, Xcat)
            print('Done in {:.3f}s'.format(time() - t0))
            return True  # True if loading succeeds
        return False  # False if loading fails

    def SaveClusterInfo(self, path):
        path = path.rstrip(r'\/')
        npy_file = path + '/labels.npy'
        if not os.path.isdir(path):
            os.makedirs(path)
        if os.path.isfile(npy_file):  # If already exists, overwrite and return False
            np.save(npy_file, self.labels)
            return False
        else:
            np.save(npy_file, self.labels)
            return True
        
    def SaveClusterMap(self, path):
        path = path.rstrip(r'\/')
        map_file = path + '/labels.jpg'

        colorTable = self.rng.rand(self.n_clusters, 3)

        clusterMap = colorTable[self.labels].reshape(self.size_y, self.size_x, 3)
        WriteImg(map_file, clusterMap, False)








