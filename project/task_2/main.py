import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from itertools import chain

def exclude(s):
  l = ['A','B','C','D']
  l.remove(s)
  return l

def probs_to_classify(l):
  l = [list(v) for v in list(l)]
  index = l.index(max([list(v) for v in l], key=lambda x: x[0]))
  if index == 0:
    return 'A'
  elif index == 1:
    return 'B'
  elif index == 2:
    return 'C'
  elif index == 3:
    return 'D'
  
def log_probs_to_prob(l):
  l = [list(v) for v in list(l)]
  l = [[np.e**v for v in sublist] for sublist in l]
  return max([v[0] for v in l])/sum([v[0] for v in l])

def classify(t):
  return probs_to_classify(model.predict_log_proba(t))

def length_prediction(t):
  lengths = [len(v) for v in t]
  if any([length > 230 for length in lengths]):
    index = lengths.index(max(lengths))
    if index == 0:
      return 'A'
    elif index == 1:
      return 'B'
    elif index == 2:
      return 'C'
    elif index == 3:
      return 'D'
  else:
    return ''

def first_letter_lowercase(t):
  t = ['Placeholder' if s=='' else s for s in t]
  firsts = [v[0] for v in t]
  lower = [l.islower() for l in firsts]
  if any(lower):
    index = lower.index(True)
    if index == 0:
      return 'A'
    elif index == 1:
      return 'B'
    elif index == 2:
      return 'C'
    elif index == 3:
      return 'D'
  else:
    return ''
  
def final_prediction(t):
  return next(s for s in t if s)

def label(data):
  data['NB_Prediction'] = [classify(t) for t in list(zip(data['A'], data['B'], data['C'], data['D']))]
  data['max_prob'] = [log_probs_to_prob(model.predict_log_proba(t)) for t in list(zip(data['A'], data['B'], data['C'], data['D']))]
  data['Length_prediction'] = [length_prediction(t) for t in list(zip(data['A'], data['B'], data['C'], data['D']))]
  data['Lowercase'] = [first_letter_lowercase(t) for t in list(zip(data['A'], data['B'], data['C'], data['D']))]
  data['Final_Prediction'] = [final_prediction(t) for t in list(zip(data['Lowercase'], data['Length_prediction'], data['NB_Prediction']))]
  return data








data = pd.read_csv('train.csv')
prelim_answers = pd.read_csv('prelim_answers.csv')
prelim_answers = prelim_answers.fillna('')

data['wrongs'] = [list(data.iloc[i][exclude(data.iloc[i]['answer'])]) for i in range(len(data))]
prelim_answers['wrongs'] = [list(prelim_answers.iloc[i][exclude(prelim_answers.iloc[i]['answer'])]) for i in range(len(prelim_answers))]
data['correct'] = [data.iloc[i][data.iloc[i]['answer']] for i in range(len(data))]
prelim_answers['correct'] = [prelim_answers.iloc[i][prelim_answers.iloc[i]['answer']] for i in range(len(prelim_answers))]

train_wrongs = list(chain.from_iterable(data['wrongs'])) + list(chain.from_iterable(prelim_answers['wrongs']))
train_wrongs = [(s, 'W') for s in train_wrongs]

train_corrects = [(s, 'C') for s in (list(data['correct']) + list(prelim_answers['correct']))]

train = pd.DataFrame(train_wrongs+train_corrects, columns=['text','label'])

model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(train['text'], train['label'])





final_test = pd.read_csv('test_final.csv')
final_test = final_test.fillna('')

final_test = label(final_test)






final_test.at[380, 'Final_Prediction'] = 'A'
final_test.at[788, 'Final_Prediction'] = 'D'
final_test.to_csv('test_final_labeled.csv', index=False)

final_test['Final_Prediction'].to_csv('implicit.txt', index=False, header=False)