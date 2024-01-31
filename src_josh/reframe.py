import os
import pandas as pd
import numpy as np
from random import shuffle
import time
import argparse
from random import shuffle
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv

if os.environ.get('ENV') != 'production':
    load_dotenv()
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_path', type=str, default='data/reframing_dataset.csv', help='Path to the training data')
    parser.add_argument('--test_path', type=str, default='data/sample_test.csv', help='Path to the test data')
    parser.add_argument('--output_path', type=str, default='data/sample_test_output.csv', help='Path to the output file')
