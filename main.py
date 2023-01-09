# main.py
#   run cbirs with in parallel
# by: Noah Syrkis

# imports
from multiprocessing import Pool
from cbir import run


# main
def main():
    params = []  # 32 different parameters we wanna try
    with Pool(32) as p:
        p.map(run, params)