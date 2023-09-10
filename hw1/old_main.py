import sys
import pandas as pd
import numpy as np

def entropy(target):
    elements, counts = np.unique(target['class'], return_counts = True)
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts))for i in range(len(elements))])
    return entropy

def information_gain(target, col):
    H = entropy(target)
    elements = np.unique(target[col])
    total = []
    for el in elements:
        class_df = target[target[col] == el]
        weight = len(class_df)/len(target)
        total.append(weight*entropy(class_df))
    IG = H - sum(total)
    return IG

def ascii_tree(df, prefix=""):
    # all attributes are the same
    if all(df.nunique() == 1):
        print(df.iloc[0]['class'])
        return
    # all the classes are the same
    elif len(df['class'].unique()) == 1:
        print(
            df['class'].unique()[0])
        return

    # Your code here
    info_gains = {col:information_gain(df, col) for col in df.columns if col != 'class'}
    best_attr = max(info_gains, key=info_gains.get)

    # Go through all the elements of that attribute
    for value in [0,1,2]:
        print(f'{prefix}{best_attr} = {value} :')

def main(train_file, test_file):
    train = pd.read_csv(train_file, sep='\t')
    test = pd.read_csv(test_file, sep='\t')
    ascii_tree(train)

if __name__ == "__main__":
    # train_file = sys.argv[1]
    # test_file = sys.argv[2]
    train_file = "train/train.dat"
    test_file = "test/test.dat"
    main(train_file, test_file)
