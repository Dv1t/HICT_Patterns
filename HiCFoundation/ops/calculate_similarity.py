# This script is to calculate the similarity between two Hi-C using a pre-trained reproducibility model.

import os
import sys
import numpy as np
import pickle
from collections import defaultdict

input_pickle1 = sys.argv[1]
input_pickle2 = sys.argv[2]

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

input1 = load_pickle(input_pickle1)
input2 = load_pickle(input_pickle2)

def find_key(chr,loc,key_list):
    """
    Find the key in the list of keys that contains the given chromosome and location.
    """
    key1 = chr+":"+loc
    if key1 in key_list:
        return key1
    key1 = "chr"+chr+":"+loc
    if key1 in key_list:
        return key1
    key1 = chr+"_"+chr+":"+loc
    if key1 in key_list:
        return key1
    key1 = "chr"+chr+"_chr"+chr+":"+loc
    if key1 in key_list:
        return key1
    return None

def calculate_similarity(input1, input2):
    """
    Calculate the similarity between two Hi-C matrices using a pre-trained reproducibility model.
    """
    similarity_dict = defaultdict(list)
    for key in input1.keys():
        #1_1:1960,1960 format of key
        split_chromosome = key.split(":")[0]
        split_loc = key.split(":")[1]
        combine_key = split_chromosome + ":" + split_loc
        chr = split_chromosome.split("_")[0]
        chr = chr.replace("chr","")
        if combine_key not in input2.keys():
            combine_key = find_key(chr,split_loc,input2.keys())
            if combine_key is None:
                continue
        
        embedding1 = input1[key]
        embedding2 = input2[combine_key]
        # Calculate the similarity between the two embeddings
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        if np.isnan(similarity):
            continue
        similarity_dict[chr].append(similarity)
    #ignore chrY, chrM, Un, Alt cases
    similarity_list=[]
    for chrom in similarity_dict:
        if "Y" in chrom or "M" in chrom or "Un" in chrom or "Alt" in chrom:
            continue
        mean_val = np.mean(similarity_dict[chrom])
        similarity_list.append(mean_val)
    similarity = np.mean(similarity_list)
    return similarity

similarity = calculate_similarity(input1, input2)
print("The reproducibility score between the two Hi-C is: ", similarity)
    