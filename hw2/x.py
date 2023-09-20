import sys
import pandas as pd
import numpy as np

def main(train_file, test_file):
    # BEGIN SCRIPT
    data = pd.read_csv(train_file, sep="\t")
    test_data = pd.read_csv(test_file, sep="\t")
    attribute_names = data.columns[:-1].tolist()
    target = data.columns[-1]
    classes = data[target].unique().tolist()
    attribute_values = sorted(data[attribute_names].stack().unique())
    # Initialize an empty DataFrame to hold the conditional probabilities
    conditionals_df = pd.DataFrame(
        index=pd.MultiIndex.from_product([classes, attribute_names], names=['Class', 'Attribute']),
        columns=attribute_values
    ).fillna(0)  # Initialize with zeros

    for class_label in classes:
        subset_data = data[data[target] == class_label]
        for attr in attribute_names:
            for attr_label in attribute_values:
                # P(attr=attr_label|class=class_label)
                conditionals_df.loc[(class_label, attr), attr_label] = len(subset_data[subset_data[attr]==attr_label])/len(subset_data)
    
    # P(class=class_label)
    class_0 = len(data[data[target] == 0])/len(data)
    class_1 = 1-class_0

    


    


    
    


    

if __name__ == "__main__":
    # train_file = sys.argv[1]
    # test_file = sys.argv[2]
    train_file = 'train/train.dat' 
    test_file = 'test/test.dat'
    main(train_file, test_file)
