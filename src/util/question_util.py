from dataclasses import dataclass

import numpy as np
from numpy import ndarray


@dataclass
class QuestionUtil:
    @staticmethod
    def gen_random_tsp(num_cities: int) -> tuple[ndarray, ndarray]:
        rng = np.random.default_rng()

        # 座標
        locations: ndarray = rng.random(size=(num_cities, 2))

        # 距離行列
        x = locations[:, 0]
        y = locations[:, 1]
        distances: ndarray = np.sqrt(
            (x[:, np.newaxis] - x[np.newaxis, :]) ** 2
            + (y[:, np.newaxis] - y[np.newaxis, :]) ** 2
        )

        return locations, distances
