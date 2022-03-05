import pandas as pd
import numpy as np
from use_4 import *

def get_question_bank_df():
    df = pd.read_csv("question_bank.csv")
    return df

def get_questions():
    df = pd.read_csv("question_bank.csv")
    questions = np.array(df["question"].values, dtype=str)
    return questions

def generate_embedded_encoding():
    model = USE4()
    questions_bank = get_questions()
    questions_bank_embedded = np.array(model(questions_bank))
    np.save("questions_bank_encoded.npy", questions_bank_embedded)