import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys
import pandas as pd
import numpy as np

def predict_instance(row, class_priors, conditionals_df):
    best_class, best_log_prob = None, -float('inf')
    attribute_names = conditionals_df.loc[0].index.tolist()

    for class_label in class_priors.index:
        log_prob = np.log(class_priors[class_label])
        for attr in attribute_names:
            value = np.log(conditionals_df.loc[(class_label, attr), row[attr]])
            log_prob += value
        if log_prob > best_log_prob:
            best_class = class_label
            best_log_prob = log_prob
    
    return best_class

def main(train_file, test_file):
    data = pd.read_csv(train_file, sep="\t")
    test_data = pd.read_csv(test_file, sep="\t")
    attribute_names = data.columns[:-1].tolist()
    target = data.columns[-1]
    classes = data[target].unique().tolist()
    attribute_values = sorted(data[attribute_names].stack().unique())
    # Initialize an empty DataFrame to hold the conditional probabilities
    conditionals_df = pd.DataFrame(
        index=pd.MultiIndex.from_product([classes, attribute_names], names=['Class', 'Attribute']),
        columns=attribute_values,
        dtype='float64'
    ).fillna(0)  # Initialize with zeros

    for class_label in classes:
        subset_data = data[data[target] == class_label]
        for attr in attribute_names:
            value_counts = subset_data[attr].value_counts()
            total_counts = len(subset_data)
            for value, count in value_counts.items():
                # P(attr=value|class=class_label)
                conditionals_df.loc[(class_label, attr), value] = count/total_counts
    
    # P(class=class_label)
    total_counts = len(data)
    class_priors = data[target].value_counts()/total_counts

    for class_label in class_priors.index:
        print(f"P(class={class_label})={class_priors[class_label]:.2f}", end=' ')
        for attribute in attribute_names:
            for value in attribute_values:
                conditional_prob = conditionals_df.loc[(class_label, attribute), value]
                print(f"P({attribute}={value}|{class_label})={conditional_prob:.2f}", end=' ')
        print()
    
    predict_instance(data.iloc[0], class_priors, conditionals_df)
    data['Predicted'] = data.apply(lambda row: predict_instance(row, class_priors, conditionals_df), axis=1)

    accuracy_train = sum(data['Predicted'] == data[target])/len(data)

    test_data['Predicted'] = test_data.apply(lambda row: predict_instance(row, class_priors, conditionals_df), axis=1)

    accuracy_test = sum(test_data['Predicted'] == test_data[target])/len(test_data)

    accuracy_train_percent = accuracy_train * 100
    accuracy_test_percent = accuracy_test * 100

    print()
    print(f"Accuracy on training set ({len(data)} instances): {accuracy_train_percent:.2f}%")
    print()
    print(f"Accuracy on test set ({len(test_data)} instances): {accuracy_test_percent:.2f}%")
    print()


if __name__ == "__main__":
    # train_file = sys.argv[1]
    # test_file = sys.argv[2]
    train_file = 'train/train.dat' 
    test_file = 'test/test.dat'
    main(train_file, test_file)