from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt


@dataclass
class ShowUtil:
    @staticmethod
    def show_plot(locations: np.ndarray):
        plt.figure(figsize=(7, 7))
        plt.xlabel("x")
        plt.ylabel("y")
        plt.scatter(locations[:, 0], locations[:, 1])
        plt.show()

    @staticmethod
    def show_route(
        route: np.ndarray, distances: np.ndarray, locations: np.ndarray, num_cities: int
    ):
        path_length = sum(
            [distances[route[i]][route[i + 1]] for i in range(num_cities)]
        )

        x = [i[0] for i in locations]
        y = [i[1] for i in locations]
        plt.figure(figsize=(7, 7))
        plt.title(f"path length: {path_length}")
        plt.xlabel("x")
        plt.ylabel("y")

        for i in range(num_cities):
            r = route[i]
            n = route[i + 1]
            plt.plot([x[r], x[n]], [y[r], y[n]], "b-")
        plt.plot(x, y, "ro")
        plt.show()

        return path_length
