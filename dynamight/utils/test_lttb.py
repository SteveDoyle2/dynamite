import numpy as np
import matplotlib.pyplot as plt
from dynamight.utils.lttb import largest_triangle_three_buckets

def main():
    data = np.loadtxt('source.csv', delimiter=',')
    data_list = data.tolist()

    plt.plot(data[:, 0], data[:, 1])
    for threshold in [200, 500, 1000, 2000]:
        sampled = largest_triangle_three_buckets(data_list, threshold)
        sampled2 = np.array(sampled)
        plt.plot(sampled2[:, 0], sampled2[:, 1], label=f'threshold={threshold}')
    plt.legend()
    plt.show()


main()

