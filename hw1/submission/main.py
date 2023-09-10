import sys
import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
from math import log



def main(train_file, test_file):
    # DEFINE FUNCTIONS
    def list_to_dist(l):
        counts = [l.count(item) for item in set(l)]
        total = sum(counts)
        return [x/total for x in counts]

    def entropy(dist):
        if len(dist) == 1:
            return 0
        result = [-1*probability*log(probability, 2) for probability in dist]
        return sum(result)

    def weighted_avg_entropy(lists):
        return sum([len(l)*entropy(list_to_dist(l)) for l in lists])/sum([len(l) for l in lists])

    class Node:
        def __init__(self, df):
            self.df = df
            self.children = {}
            self.split_attribute = ""

        def is_pure(self):
            return entropy(list_to_dist(list(self.df[target]))) == 0

        def majority(self):
            targets = list(self.df[target])
            rank = [0,1,2]
            rank.sort(key = lambda x: target_counts[x], reverse=True)
            rank.sort(key = lambda x: targets.count(x), reverse=True)
            return rank[0]


    def split_by_column(data, column):
        return {
            value:Node(data[data[column] == value].drop(column, axis=1)) for value in [0,1,2]
        }

    def splits_by_all_columns(data):
        return {
            column:[list(node.df[target]) for node in split_by_column(data, column).values()] for column in data.columns if column != target
        }

    def entropy_by_all_columns(data):
        return {
            item[0]:weighted_avg_entropy(item[1]) for item in splits_by_all_columns(data).items()
        }

    def build_tree(root):
        if root.is_pure() or len(root.df.columns) == 1:
            return
        else:
            purest_attribute_to_split = (min(entropy_by_all_columns(root.df).items(), key = lambda x: x[1]))[0]
            root.split_attribute = purest_attribute_to_split
            root.children = split_by_column(root.df, purest_attribute_to_split)
            for child in root.children.values():
                build_tree(child)

    def print_tree(node, depth, first=False):
        if node.children == {}:
            print(" " + str(node.majority()), end="")
        else:
            for idx, child in enumerate(node.children.items()):
                if not first or idx != 0:
                    print()
                print(depth*"| ", end="")
                print(node.split_attribute, end=" = ")
                print(str(child[0]) + " :", end="")
                print_tree(child[1], depth+1)

    def list_to_dict(l, cols):
        cols.remove(target)
        if len(l) ==len(cols):
            return {
                cols[i]:l[i] for i in range(len(l))
            }
        else:
            print(len(l))
            print(len(cols))
            return None

    def traverse(node, row):
        if node.children == {}:
            return node.majority()
        else:
            return traverse(node.children[row[node.split_attribute]], row)
    
    def predict(tree, data):
        rows = [list_to_dict(l, list(data.columns)) for l in data.drop(target, axis=1).values.tolist()]
        return [traverse(tree, row) for row in rows]
    
    def accuracy(tree, data):
        data['Predicted'] = predict(tree, data)
        return round(100*sum(data[target] == data['Predicted'])/len(data), 1)


    # BEGIN SCRIPT
    data = pd.read_csv(train_file, sep="\t")
    test_data = pd.read_csv(test_file, sep="\t")
    target = 'class'
    attribute_values = {
        column:set(data[column]) for column in data.columns
    }
    target_counts = {x:list(data[target]).count(x) for x in [0,1,2]}

    # Build and print tree
    tree = Node(data)
    build_tree(tree)
    print_tree(tree, 0, first=True)
    print()

    # Training accuracy
    training_accuracy = accuracy(tree, data)
    print()
    print("Accuracy on training set (" + str(len(data)) + " instances): " + str(training_accuracy) + "%" + "\n")

    # Test accuracy
    test_accuracy = accuracy(tree, test_data)
    print("Accuracy on test set (" + str(len(test_data)) + " instances): " + str(test_accuracy) + "%" + "\n")

    # # Plot learning curve
    # learning_curve_trees = {n:Node(data.sample(n)) for n in range(100,801,100)}
    # for tree in learning_curve_trees.values():
    #     build_tree(tree)
    # learning_curve_dict ={n:accuracy(learning_curve_trees[n], test_data) for n in range(100,801,100)}
    # learning_curve_plot = pd.DataFrame(
    #     learning_curve_dict.items(), columns=['Training Set Size', 'Test Accuracy']
    # )

    # plt.figure(figsize=(10, 6))
    # sns.scatterplot(data=learning_curve_plot, x='Training Set Size', y='Test Accuracy')
    # plt.title('Learning Curve')
    # plt.xlabel('Training Set Size')
    # plt.ylabel('Test Accuracy (%)')
    # plt.grid(True)
    # plt.show()

if __name__ == "__main__":
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    # train_file = 'hw1/train/train3.dat' 
    # test_file = 'hw1/test/test3.dat'
    # train_file = 'hw1/train/train2.dat' 
    # test_file = 'hw1/test/test2.dat'
    main(train_file, test_file)
