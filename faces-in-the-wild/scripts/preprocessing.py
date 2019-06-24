"""
Preprocess the training kinship relationships and the faces images

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class FacialData(object):
    def __init__(self):
        self._train_relationships_df = pd.read_csv('../data/train_relationships.csv')
        self._index_by_family()

    def _index_by_family(self):
        cols = ['family', 'p1', 'p2']  # new DF with family index
        self._family_df = pd.DataFrame(columns=cols)
        for i in range(len(self._train_relationships_df)):
            p1 = self._train_relationships_df.iloc[i].p1.split('/')
            p2 = self._train_relationships_df.iloc[i].p2.split('/')
            self._family_df.loc[i] = [p1[0], p1[1], p2[1]]  # family, first person, kinship relation person

    def print_info(self):
        print("Basic description:\n")
        print(self._train_relationships_df.describe())
        print("Head:\n")
        print(self._train_relationships_df.head())
        print("Total number of families: ", self._family_df.family.nunique())
        print(self._family_df.family.count())
        print("One family has 75 relationships: ", self._family_df.family.value_counts().idxmax())

    def plot_relationship_histogram(self):
        f = self._family_df.family.unique()
        number_of_relations = []
        for family in f:
            number_of_relations.append(self._family_df.loc[self._family_df.family == family].values.shape[0])
        plt.hist(number_of_relations, bins=100)  # most families have 10 relations, there are entries for 70+ relations
        print("Checking number of entries: ", len(self._train_relationships_df), np.cumsum(number_of_relations)[-1])
        plt.show()

def main():
    fd = FacialData()
    fd.print_info()
    # fd.plot_relationship_histogram()


if __name__ == '__main__':
    main()