#!/usr/bin/env python
# coding: utf-8


import numpy as np
import cv2
import os
import json
from scipy import ndimage
from sklearn.cluster import KMeans
from typing import List, T
from sklearn.model_selection import train_test_split


dataset_path = 'caltech-101/101_ObjectCategories'
test_size = 0.2



def get_file_paths(dataset_path : str, category : str, test_size : float) -> List[str]:
    """ 
    returns names of files for test and train datasets of particular category
    """
    files_paths = []
    for root, dirs, files in os.walk(os.path.join(dataset_path, category)):
        for name in files:
            files_paths.append(os.path.join(root, name))
    train_files, test_files = train_test_split(files_paths, test_size=test_size)
    return train_files, test_files


def load_images(files_paths : List[str]) -> List[np.ndarray]:
    images = []
    for file in files_paths:
        img = cv2.imread(file,0)
        images.append(img)
    return images

        
def sift_features(images):
    descriptor_dict = {}
    descriptor_list = []
    sift = cv2.SIFT_create()
    for category, cat_images in images.items():
        features = []
        for img in cat_images:
            kp, des = sift.detectAndCompute(img,None)
            if (des is not None):
                descriptor_list.extend(des)
                features.append(des)
        descriptor_dict[category] = features
    return [descriptor_list, descriptor_dict]


def kmeans(k, descriptor_list):
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(descriptor_list)
    visual_words = kmeans.cluster_centers_ 
    return visual_words
  

def find_index(feature : np.ndarray, centers : np.ndarray, dist_fn : T):
    # returns an index of the visual word that is the most similar to a feature 
    minimum = 0
    start_min = True 
    min_index = 0 
    for idx, center in enumerate(centers): 
        distance_c = dist_fn(center, feature)
        if start_min:
            minimum = distance_c
            start_min = False 
        else: 
            minimum = min(minimum, distance_c)
            if minimum == distance_c:
                min_index = idx

    return min_index


# Takes 2 parameters: 
# - a dictionary that holds the descriptors that are separated class by class 
# - an array that holds the central points (visual words) of the k means clustering
# Returns a dictionary that holds the histograms for each images that are separated class by class. 
def image_class(all_bovw, centers, dist_fn):
    bovw = {}
    for key,value in all_bovw.items():
        category = []
        for img in value:
            histogram = np.zeros(len(centers))
            for feature in img:
                ind = find_index(feature, centers, dist_fn)
                histogram[ind] += 1
            category.append(histogram)
        bovw[key] = category
    return bovw


def save_to_json(file_name, bovw):
    bovw_list = {cat : list(bovw[cat][0]) for cat in bovw.keys()}
    json_object = json.dumps(bovw_list)
    with open(file_name, 'w') as outfile:
        outfile.write(json_object)


# 1-NN algorithm. We use this for predict the class of test images.
# Takes 2 parameters
# - images : feature vectors of train images 
# - tests : feature vectors of test images
# Returns an array that holds number of test images, number of correctly predicted images and records of class based images respectively
def knn(images, tests, dist_fn):
    num_test = 0
    correct_predict = 0
    class_based = {}
    
    for test_category, test_histograms_list in tests.items():
        class_based[test_category] = [0, 0] # [correct, all]
        for test_histogram in test_histograms_list:
            min_start = True 
            for train_category, train_histograms_list in images.items():
                for train_histogram in train_histograms_list:
                    if min_start:
                        minimum = dist_fn(test_histogram, train_histogram)
                        pred = train_category
                        min_start = False
                    else:
                        dist = dist_fn(test_histogram, train_histogram)
                        if(dist < minimum):
                            minimum = dist
                            pred = train_category
            
            if(test_category == pred):
                correct_predict += 1
                class_based[test_category][0] += 1
            num_test += 1
            class_based[test_category][1] += 1

    return [num_test, correct_predict, class_based]
   

# Calculates the average accuracy and class based accuracies.  
def accuracy(results, dist_fn):
    avg_accuracy = (results[1] / results[0]) * 100
    out_str = ""
    out_str += "Average accuracy: %" + str(avg_accuracy) + '\n'
    out_str += "\nClass based accuracies: \n"
    for key,value in results[2].items():
        acc = (value[0] / value[1]) * 100
        out_str += "\n" + key + " : %" + str(acc)
    with open(f"stds/{k}_{dist_fn.func_name}", 'w') as f:
        f.write(out_str)


def run(params):
    k = params['k']
    dist_fn = params['dist']

    categories = []
    for root, dirs, files in os.walk(dataset_path):
        for dir in dirs:
            
            for cat_root, cat_dirs, cat_files in os.walk(os.path.join(dataset_path, dir)):
                
                if len(cat_files) >= 100:
                    categories.append(dir)
                    print(dir, len(cat_files))

    # creating train and test dataset s 
    train_images = {}
    test_images = {}
    for cat in categories:
        train_files, test_files = get_file_paths(dataset_path, cat, test_size)
        train_images[cat] = load_images(train_files)
        test_images[cat] = load_images(test_files)

    descriptor_list, all_bovw_feature = sift_features(train_images) 
    _, test_bovw_feature = sift_features(test_images)
    # descriptor_list is needed for creating clustering centers, so we only take them from train dataset
    # bovw_feature for train and test are needed for classification of the image 

    # Returns an array that holds central points (visual words)
    visual_words = kmeans(k, descriptor_list) 
    
    # Creates histograms for train data    
    bovw_train = image_class(all_bovw_feature, visual_words, dist_fn) 
    # Creates histograms for test data
    bovw_test = image_class(test_bovw_feature, visual_words, dist_fn)

    save_to_json('bovw_test.json', bovw_test)
    save_to_json('bovw_train.json', bovw_train)
 
    # Call the knn function    
    results_bowl = knn(bovw_train, bovw_test, dist_fn) 
        
    # Calculates the accuracies and write the results to the console.       
    accuracy(results_bowl, dist_fn) 

    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(6, 2)
    fig.set_size_inches(20, 20)
    for ax, cat in zip(axes, categories):
        ax[0].hist(bovw_train[cat], bins=k)
        ax[0].set_title(f'{cat} train histogram')
        ax[1].hist(bovw_test[cat], bins=k)
        ax[1].set_title(f'{cat} test histogram')

    plt.savefig(f'plots/{k}_{dist_fn.func_name}_histograms.png', dpi=300)

