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
    params = [{'k' : k, 'dist': dist} for k in [250, 500, 1000] for dist in [d.cosine, d.euclidean, d.minkowski, d.chebyshev]]
    run(params[0])
    exit()
    print(params)
    with Pool(2) as p:
        p.map(run, params)

if __name__ == '__main__':
    main()