import argparse
import csv
import json
import os
import statistics
import sys
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def get_num_matches(results: list, baselines: list) -> Optional[int]:
    """
    Input: 
        - results: List of items retrieved for endpoint query.
        - baseline: List of items relevant to the endpoint query.
    Output:
        - num_hits: Number of retrieved items that match an item in the baseline (relevant) list.
    """
    num_hits = 0
    exhausted = set()

    for r in results:
        # Lowercase the returned label for case-insensitive substring matching
        eid = r["@id"].lower()
        # Accept if ANY of the acceptable ids was found
        for a in baselines:
            if eid == a['@id'].lower() and not a['@id'].lower() in exhausted:
                num_hits += 1  # 1-based: first result in list is rank 1
                exhausted.add(a['@id'].lower())

    return num_hits # correct answer not found in the top-k results returned by the API

def json_file_to_list(filepath):
    data = []
    search_terms = []
    with open(filepath, 'r') as f:
        for line in f:
            res_dict = {}
            json_line = json.loads(line)
            other_props = {}
            for k,v in json_line.items():
                if k == 'search_term':
                    res_dict[k] = v
                    search_terms.append(v)
                else:
                    other_props[k] = v
            res_dict['other_props'] = other_props
            data.append(res_dict)
    return search_terms, data

def extract_matches(search_term, list_dict):
    for v in list_dict:
        if v["search_term"] == search_term:
            return v["other_props"]["matches"]

def main():
    owl_search_res = "owldb_class_search_chiro_mondo_exact_match_false_at_least_n_minus_one.jsonl"
    bp_search_res = "bp_class_search_chiro_mondo_exact_match_false_at_least_n_minus_one.jsonl"

    search_terms, owl_dict_list = json_file_to_list(owl_search_res)
    _, bp_dict_list = json_file_to_list(bp_search_res)

    all_ep_precisions = []
    all_ep_recalls = []
    for search_term in search_terms:
        owl_dict_matches = extract_matches(search_term, owl_dict_list)
        bp_dict_matches = extract_matches(search_term, bp_dict_list)
        num_matches = get_num_matches(owl_dict_matches, bp_dict_matches)
        print(f"Number of common entities: {num_matches}")
        all_ep_precisions.append(round(num_matches/len(bp_dict_matches), 3))
        all_ep_recalls.append(round(num_matches/len(owl_dict_matches), 3))

    prec_q_05_at_k = round(np.quantile(all_ep_precisions, 0.25), 3)
    prec_q_50_at_k = round(np.quantile(all_ep_precisions, 0.5), 3)
    prec_q_95_at_k = round(np.quantile(all_ep_precisions, 0.95), 3)
    recall_q_05_at_k = round(np.quantile(all_ep_recalls, 0.25), 3)
    recall_q_50_at_k = round(np.quantile(all_ep_recalls, 0.5), 3)
    recall_q_95_at_k = round(np.quantile(all_ep_recalls, 0.95), 3)

    avg_precision_at_k = round(sum(all_ep_precisions)/len(all_ep_precisions), 3)
    avg_recall_at_k = round(sum(all_ep_recalls)/len(all_ep_recalls), 3)
    print(f"    {'Average precision@k':<12}{avg_precision_at_k:>28}")
    print(f"    {'5% quantile precision@k':<5}{prec_q_05_at_k:>22}")
    print(f"    {'50% quantile precision@k':<6}{prec_q_50_at_k:>22}")
    print(f"    {'95% quantile precision@k':<6}{prec_q_95_at_k:>23}")
    print(f"    {'Average recall@k':<12}{avg_recall_at_k:>31}")
    print(f"    {'5% quantile recall@k':<5}{recall_q_05_at_k:>25}")
    print(f"    {'50% quantile recall@k':<6}{recall_q_50_at_k:>26}")
    print(f"    {'95% quantile recall@k':<6}{recall_q_95_at_k:>26}")

if __name__ == "__main__":
    main()