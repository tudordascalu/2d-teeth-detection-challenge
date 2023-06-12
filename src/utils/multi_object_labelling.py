class ScoreMapper:
    def __init__(self, n_teeth=32):
        self.n_teeth = n_teeth

    def __call__(self, distance_mat, distance_mat_mean, distance_mat_std=None, distance_map_cov=None, centroids=None):
        """

        :param distance_map: np.array of shape (17, 1) featuring labels for each instance
        :param distance_map_mean: np.array of shape (17, 17) featuring mean distances between tooth-tooth pairs
        :param distance_map_std: np.array of shape (17, 17) featuring stds for tooth-tooth pair distances
        :return: np.array of shape (17, 17) denoting tooth-tooth probabilities based on distances
        """
        score_map = np.zeros((self.n_teeth, self.n_teeth, 1))
        # Navigate through all instances [0, 17]
        for i in range(self.n_teeth):
            for j in range(self.n_teeth):
                distance = distance_map[i, j]
                distance_mean = distance_map_mean[i, j]
                try:
                    if self.mode == "univariate":
                        distance_std = distance_map_std[i, j]
                        x_score = norm.pdf(distance[0], distance_mean[0], distance_std[0])
                        y_score = norm.pdf(distance[1], distance_mean[1], distance_std[1])
                        z_score = norm.pdf(distance[2], distance_mean[2], distance_std[2])
                        if math.isnan(x_score) or math.isnan(y_score) or math.isnan(z_score):
                            raise ValueError("The result of norm.pdf is NaN")
                        score = x_score + y_score + z_score
                    elif self.mode == "multivariate":
                        distance_cov = distance_map_cov[i, j]
                        score = multivariate_normal.pdf(distance, mean=distance_mean, cov=distance_cov)
                    else:
                        raise ValueError("Mode can only take values from ['univariate', 'multivariate']")
                except:
                    score = 0
                score_map[i, j] = score
        # Set scores to 0s for missing teeth
        missing_teeth = np.where((centroids == np.array([0, 0, 0])).all(axis=1))[0]
        score_map[missing_teeth] = np.ones((self.n_teeth, 1)) * -1
        score_map[:, missing_teeth] = np.ones((self.n_teeth, len(missing_teeth), 1)) * -1
        score_map[np.arange(self.n_teeth), np.arange(self.n_teeth)] = 0
        return score_map
