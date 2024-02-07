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
import spacy
from transformers import BertTokenizer, BertModel

DISTORTIONS_RFM = [
    # Same as KGL
    "all-or-nothing thinking",
    "overgeneralizing",
    "disqualifying the positive",  # focusing on negatives and ignoring the positives
    "should statements",
    "labeling",
    "personalizing",
    "magnification",
    "emotional reasoning",
    "mind reading",
    "fortune telling",
    # Extra to KGL
    "blaming",
    "comparing and despairing",
    "negative feeling or emotion",
    "catastrophizing",  # similar to magnification
]

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

if os.environ.get("ENV") != "production":
    load_dotenv()
    
def load_data(path) -> pd.DataFrame:
    data = pd.read_csv(path)
    tokenization_results = data.apply(lambda row: tokenize_data(row), axis=1)
    data['indexed_tokens'], data['segment_ids'] = zip(*tokenization_results)
    data.drop(columns=["reframe", "thinking_traps_addressed", "thought", "situation"], inplace=True)
    
    cols = ['indexed_tokens', 'segment_ids'] + [col for col in data.columns if col not in ['indexed_tokens', 'segment_ids']]
    data = data[cols]
    
    return data


def tokenize_data(row: pd.Series):
    # This function will now handle a DataFrame and a tokenizer
    situation, thought = row["situation"], row["thought"]
    

    scenario_tokens = tokenizer.tokenize(situation) + ["[SEP]"]
    thought_tokens = tokenizer.tokenize(thought) + ["[SEP]"]
    
    tokens = ["[CLS]"] + scenario_tokens + thought_tokens
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
    
    segment_ids = [0] * (len(thought_tokens) + 1)  # +1 for [CLS]
    segment_ids += [1] * len(scenario_tokens)
    
    return indexed_tokens, segment_ids


def main(
    thought: str, situation: str, training_path: str = "data/reframing_dataset_2.csv"
):
    """
    Goal is to predict the cognitive distortions in a given text input and situation.

    :param text_input: _description_
    :param situation: _description_
    :param training_path: _description_, defaults to 'data/reframing_dataset_2.csv'
    """
    data = load_data(training_path)


main(
    "Hell no, I'm hating this. Please make the pain stop.",
    "I'm a social media user on Reddit.",
)
