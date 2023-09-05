```
def entropy(target):
    elements, counts = np.unique(target['Worth Taking'], return_counts = True)
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

def is_done(df):
    # Check if all data is in the same class
    if len(df['Worth Taking'].unique()) == 1:
        print(f'make leaf node with class value as output: {df["Worth Taking"].iloc[0]}')
    # Check if all data has the same attributes in all columns
    elif all(df.nunique() == 1):
        majority = df['Worth Taking'].value_counts().idxmax()
        print(f'make leaf node with majority of class values in Y as output: {majority}')
    else:
        print('okay')

First split:
[
information_gain(id3_dataset, 'Personality'),
information_gain(id3_dataset, 'Difficulty'),
information_gain(id3_dataset, 'RMP Reviews'),
information_gain(id3_dataset, 'Easy A')
]

[0.04812703040826949,
0.24674981977443933,
0.02922256565895487,
0.15183550136234159]

[
information_gain(d_difficulty_low, 'Personality'),
information_gain(d_difficulty_low, 'RMP Reviews'),
information_gain(d_difficulty_low, 'Easy A')
]
[0.9709505944546686, 0.01997309402197489, 0.01997309402197489]

d_difficulty_low_personality_hilarious = (-)
d_difficulty_low_personality_boring = (+)


[
information_gain(d_difficulty_med, 'Personality'),
information_gain(d_difficulty_med, 'RMP Reviews'),
information_gain(d_difficulty_med, 'Easy A')
]
[0.01997309402197489, 0.5709505944546686, 0.9709505944546686]

d_difficulty_med_easya_yes = (-)
d_difficulty_med_easya_no = (-)

d_difficulty_high = (-)

```
