from transformers import MarianMTModel, MarianTokenizer
import pandas as pd
import json


##### UNCOMMENT IF NEED TO USE TRANSLATION #####
# model_name = "Helsinki-NLP/opus-mt-zh-en"
# tokenizer = MarianTokenizer.from_pretrained(model_name)
# model = MarianMTModel.from_pretrained(model_name)

def translate_chinese_to_english(text):
    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]


def cognitive_distortion_translate(
    path_to_file="data_josh/cognitive_distortion_train.jsonl",
):
    """
    Translate the Chinese text in the cognitive distortion dataset to English
    and save it as a csv file.

    line['messages'][0]['content'] describes to the model what it is. It is always:
    'Now, you are a psychologist with a good knowledge of cognitive distortion of classification predictions, 
    and please perform a multi-labeling classification task to determine whether the following posts contain cognitive distortion features 
    (other than one, partiality, psychological filtering, negative positive thinking, mind reading, prophet error, magnification, 
    emotional reasoning, modus operandi, mislabelling, culpability, culpability) and use 0 and 1. Use the Mark Down form to produce the classification results. 
    The output is in the following format: id, non-requirement, partiality, psychological filtering, negative positive thinking, mind reading, prophet error, 
    magnification, emotional reasoning, properness, mislabelling, culpability, culpability.'
    
    line['messages'][1]['content'] is the actual text that the person is feeling
    
    line['messages'][2]['content'] is the label for the distortions

    :param path_to_file: _description_, defaults to 'data_josh/cognitive_distortion_train.jsonl'
    """
    data = []
    with open(path_to_file, "r") as file:
        for line in file:
            thought = json.loads(line)["messages"][1]["content"]
            translated = translate_chinese_to_english(thought)[0]
            data.append(
                translated
            )
            print(translated)

            
    df = pd.DataFrame(data)
    df.to_csv("data_josh/cd_train_eng.csv", index=False)
    
DISTORTIONS = ["Black and White", "Overgeneralization", "Mental Filtering", "Disqualifying the Positive", "Mind Reading", "Fortune Telling", "Catastrophizing", "Emotional Reasoning", "Should Statements", "Labeling", "Self-Blame", "Blaming Others"]    
def cognitive_distortion_generate_labels(path_to_file="data_josh/cognitive_distortion_train.jsonl"):
    data = []
    with open(path_to_file, "r") as file:
        for line in file:
            values = json.loads(line)["messages"][2]["content"].split("|")
            assert len(values) == 37
            values = [int(x) for x in values[-12:]]
            assert len(values) == len(DISTORTIONS)
            data.append(values)
    df = pd.DataFrame(data, columns=DISTORTIONS)
    df.to_csv("data_josh/cd_labels.csv", index=False)

    
def concatonate_data():
    """
    Concatonate the data from the two files into one file
    """
    df1 = pd.read_csv("data_josh/cd_labels.csv")
    df2 = pd.read_csv("data_josh/cd_train_eng.csv")
    df2 = df2.rename(columns={"0": "Thought"})
    
    concatenated_df = pd.concat([df2, df1], axis=1)
    
    concatenated_df.to_csv("data_josh/cognitive_distortion.csv", index=False)

# Call the function
concatonate_data()
