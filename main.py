# main.py
#   run cbirs with in parallel
# by: Noah Syrkis

# imports
from multiprocessing import Pool
from cbir import run
# import various sklearn similarity measures
import scipy.spatial.distance as d


# main
def main():
    # 32 different parameters we wanna try (list of dist. funcs?)
    params = [(k, dist) for k in [250, 500, 1000] for dist in [d.cosine, d.city_block, d.euclidean, d.minkowski, d.chebyshev]]
    with Pool(9) as p:
        p.map(run, params)

if __name__ == '__main__':
    main()