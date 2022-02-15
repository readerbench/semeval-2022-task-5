from sklearn import metrics
import pandas as pd
import numpy as np


def check_matrix(matrix, gold, pred):
  """Check matrix dimension."""
  if matrix.size == 1:
    tmp = matrix[0][0]
    matrix = np.zeros((2, 2))
    if (pred[1] == 0):
      if gold[1] == 0:  #true negative
        matrix[0][0] = tmp
      else:  #falsi negativi
        matrix[1][0] = tmp
    else:
      if gold[1] == 0:  #false positive
        matrix[0][1] = tmp
      else:  #true positive
        matrix[1][1] = tmp
  return matrix


def compute_f1(pred_values, gold_values):
  matrix = metrics.confusion_matrix(gold_values, pred_values)
  matrix = check_matrix(matrix, gold_values, pred_values)

  #positive label
  if matrix[0][0] == 0:
    pos_precision = 0.0
    pos_recall = 0.0
  else:
    pos_precision = matrix[0][0] / (matrix[0][0] + matrix[0][1])
    pos_recall = matrix[0][0] / (matrix[0][0] + matrix[1][0])

  if (pos_precision + pos_recall) != 0:
    pos_F1 = 2 * (pos_precision * pos_recall) / (pos_precision + pos_recall)
  else:
    pos_F1 = 0.0

  #negative label
  neg_matrix = [[matrix[1][1], matrix[1][0]], [matrix[0][1], matrix[0][0]]]

  if neg_matrix[0][0] == 0:
    neg_precision = 0.0
    neg_recall = 0.0
  else:
    neg_precision = neg_matrix[0][0] / (neg_matrix[0][0] + neg_matrix[0][1])
    neg_recall = neg_matrix[0][0] / (neg_matrix[0][0] + neg_matrix[1][0])

  if (neg_precision + neg_recall) != 0:
    neg_F1 = 2 * (neg_precision * neg_recall) / (neg_precision + neg_recall)
  else:
    neg_F1 = 0.0

  f1 = (pos_F1 + neg_F1) / 2
  return f1


def extract_field(truth, submission, index):
  gold = []
  guess = []
  for key, value in truth.items():
    gold.append(value[index])
    guess.append(submission[key][index])
  return gold, guess


def compute_scoreA(truth, submission):
  gold, guess = extract_field(truth, submission, 0)
  score = compute_f1(guess, gold)
  return score


def compute_scoreB(truth, submission):
  results = []
  total_occurences = 0
  for index in range(1, 5):
    gold, guess = extract_field(truth, submission, index)
    f1_score = compute_f1(guess, gold)
    weight = gold.count(True)
    total_occurences += weight
    results.append(f1_score * weight)
  return sum(results) / total_occurences

def from_eval(filename):
  truth = []
  pred = []
  df = pd.read_csv(filename, sep="\t")
  
  truth = dict(list(df.apply(lambda x: (x.file_name,[x.misogynous,x.shaming,x.stereotype,x.objectification,x.violence]), axis=1)))
  pred = dict(list(df.apply(lambda x: (x.file_name,[x.pred_misogynous,x.pred_shaming,x.pred_stereotype,x.pred_objectification,x.pred_violence]), axis=1)))

  return compute_scoreA(truth, pred), compute_scoreB(truth, pred)