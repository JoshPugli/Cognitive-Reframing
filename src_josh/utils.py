from transformers import MarianMTModel, MarianTokenizer
import pandas as pd
import json

DISTORTIONS_CHN = [
    "Black and White",
    "Overgeneralization",
    "Mental Filtering",
    "Disqualifying the Positive",
    "Mind Reading",
    "Fortune Telling",
    "Catastrophizing",
    "Emotional Reasoning",
    "Should Statements",
    "Labeling",
    "Self-Blame",
    "Blaming Others",
]
DISTORTIONS_KGL = [
    "Personalization",
    "Labeling",
    "Fortune-telling",
    "Magnification",
    "Mind Reading",
    "All-or-nothing thinking",
    "Overgeneralization",
    "Mental filter",
    "Emotional Reasoning",
    "Should statements",
]
DISTORTIONS_RFM = [
    "disqualifying the positive",
    "blaming",
    "catastrophizing",
    "mind reading",
    "comparing and despairing",
    "all-or-nothing thinking",
    "fortune telling",
    "overgeneralizing",
    "labeling",
    "should statements",
    "personalizing",
    "emotional reasoning",
    "negative feeling or emotion",
    "magnification",
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
    df.columns = DISTORTIONS_CHN + ["Thought"]
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


# Call the function
print(len(get_distortion_list_from_refraiming_dataset()))
