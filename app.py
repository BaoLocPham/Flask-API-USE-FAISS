from flask import Flask, request, json
import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
import numpy as np
from flask_restful import Api
from flask_cors import CORS
from use_4 import *
from faiss_module import *
from utils import *

# Threshold for L2 (ecludiean) distance
THRESHOLD = 0.5

app = Flask(__name__)
model = USE4()
api = Api(app)
CORS(app)


@app.route("/")
def hello():
  return "Hello World!"

@app.route("/check", methods=["POST"])
def checkQuestion():
  """
  This function is used to check whether the question is in the question bank.
  """
  questions = json.loads(request.data)
  questions = questions['questions']
  questions_embedded = np.array(model(questions))
  questions_bank = get_questions()
  questions_bank_df = get_question_bank_df()
  try: # check if the question bank has been generated
    questions_bank_embedded = np.load("question_bank_encoded.npy")
  except: # if not, generate it
    questions_bank_embedded = np.array(model(questions_bank))
    np.save("question_bank_encoded.npy", questions_bank_embedded)
  # initialize faiss
  index = FAISS(questions_bank_embedded)
  # add questions to faiss
  index.add(questions_bank_embedded)
  # search the question in the question bank
  D, I = index.search(questions_embedded, top_k=5)
  results = questions_bank_df.iloc[I[:,0]].values.tolist()
  # if distance larger than the threshold then there is no duplicate question
  results = [results[x][0] if D[x][0] < THRESHOLD else "" for x in range(len(questions))]
  return {"results": results}

@app.route("/add/question", methods=["POST"])
def addQuestion():
  """This function is used to add a new question to the question bank."""
  questions = json.loads(request.data)
  questions = questions['questions']
  # append to question_bank.csv
  print("----Add new question to question_bank.csv----")
  question_bank = get_question_bank_df()
  question_to_add = pd.DataFrame({"question":questions})
  question_bank = question_bank.append(question_to_add, ignore_index=True)
  question_bank.to_csv("question_bank.csv", index=False)
  # append to question_bank_encoded.npy
  print("----Add new question to question_bank_encoded.npy----")
  questions_bank_embedded = np.load("question_bank_encoded.npy")
  questions_to_add_embedded = np.array(model(questions))
  questions_bank_embedded = np.append(questions_bank_embedded, questions_to_add_embedded, axis=0)
  np.save("question_bank_encoded.npy", questions_bank_embedded)

  return {"results": "success"}
    
if __name__ == "__main__":
    app.run()