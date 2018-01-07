import numpy as np

class Pyramid_Kernel:

    def __init__(self, k, pyramid_levels):
        self.k = k
        self.pyramid_levels = pyramid_levels


    def intersection_kernel(self, X, Y):
        n_samples_1, n_features = X.shape
        n_samples_2, _ = Y.shape

        intersection = np.zeros((n_samples_1, n_samples_2))

        for i in range(n_samples_1):
            for j in range(n_samples_2):
                intersection[i, j] = np.minimum(X[i, :], Y[j, :]).sum()

        return intersection

    def pyramid_kernel(self, X, Y):

        #start with finer levels, new_pyramid_levels = [[4, 4], [2, 2], [1, 1]]
        pyramid_levels = self.pyramid_levels.reverse()
        X = np.fliplr(X)
        Y = np.fliplr(Y)

        last_index = 0
        previous_level_intersection = 0
        intersection = 0

        for i in range(0,len(settings.pyramid_levels)):

            num_partitions = pyramid_levels[i][0] * pyramid_levels[i][1]

            this_level_intersection = intersection_kernel(
                                X[:,last_index:last_index + self.k*num_partitions],
                                Y[:,last_index:last_index + self.k*num_partitions])

            if i == 0:
                intersection = this_level_intersection
            else:
                intersection += 2**-i * (this_level_intersection - previous_level_intersection)

            previous_level_intersection = this_level_intersection
            last_index += settings.codebook_size * num_partitions

        return intersection