from transformers import MarianMTModel, MarianTokenizer
import pandas as pd
import json

DISTORTIONS_CHN_UNTRANSLATED = [
    "非此即彼", # either/or
    "以偏概全", # generalize from partial to complete
    "心理过滤", # mental filter
    "否定正面思考", # Negate positive thinking
    "读心术", # Mind reading
    "先知错误", # prophetic error
    "放大", # enlarge
    "情绪化推理", # emotional reasoning
    "应该式", # should form
    "乱贴标签", # Tags indiscriminately
    "罪责归己", # Blame yourself
    "罪责归他", # The blame lies with him
]
DISTORTIONS_CHN_TRANSLATED = [
    "All-or-nothing thinking",
    "Overgeneralization",
    "Mental Filtering", # Combine this and Disqualifying the Positive into mental filter
    "Disqualifying the Positive", # Combine this and mental filtering into mental filter
    "Mind Reading",
    "Fortune Telling",
    "Magnification",
    "Emotional Reasoning",
    "Should Statements",
    "Labeling",
    "Personalization",
    
    "Blaming Others", 
]

DISTORTIONS_KGL = [
    "All-or-nothing thinking",
    "Overgeneralization",
    "Mental filter",  # focusing on negatives and ignoring the positives
    "Should statements",
    "Labeling",
    "Personalization",
    "Magnification",
    "Emotional Reasoning",
    "Mind Reading",
    "Fortune-telling",
]

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


##### UNCOMMENT IF NEED TO USE TRANSLATION #####
# model_name = "Helsinki-NLP/opus-mt-zh-en"
# tokenizer = MarianTokenizer.from_pretrained(model_name)
# model = MarianMTModel.from_pretrained(model_name)


def translate_chinese_to_english(text):
    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]


def translate_tsv(path_to_file="data_unparsed/cognitive_distortion_val_BERT.tsv"):
    df = pd.read_csv(path_to_file, sep="\t")
    all_translated = []
    df.columns = DISTORTIONS_CHN_TRANSLATED + ["Thought"]
    for row in df["Thought"]:
        translated = translate_chinese_to_english(row)[0]
        all_translated.append(translated)
        print(translated)

    df["Thought"] = all_translated

    column_to_move = df.pop("Thought")
    df.insert(0, "Thought", column_to_move)

    df.to_csv("data_unparsed/cognitive_distortion_val.csv", index=False)


def get_distortion_list_from_kaggle_dataset():
    df = pd.read_csv("data/kaggle/Annotated_data.csv")
    l1 = df["Dominant Distortion"].unique().tolist()
    l2 = df["Secondary Distortion (Optional)"].unique().tolist()

    l1 = set(l1)
    l2 = set(l2)

    return list(l1.union(l2))


def get_distortion_list_from_refraiming_dataset():
    df = pd.read_csv("data/reframing_dataset.csv")
    lst = df["thinking_traps_addressed"].unique().tolist()
    s = set()
    for item in lst:
        item = item.split(",")
        for i in item:
            s.add(i.strip())
    return list(s)


def get_distortion_list_from_chinese_dataset():
    f = open("data/data_raw/cognitive_distortion_train.jsonl", "r")
    line = f.readline()
    distortions = [
        x.strip()
        for x in json.loads(line)["messages"][2]["content"]
        .split("/n")[0]
        .split("|")[1:]
    ]

    return distortions


# Call the function
print(get_distortion_list_from_chinese_dataset())
print(len(get_distortion_list_from_chinese_dataset()))
