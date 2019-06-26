"""
Preprocess the training kinship relationships and the faces images

"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class FacialData(object):
    def __init__(self):
        self._train_relationships_df = pd.read_csv('../data/train_relationships.csv')
        self._root_dir = '../data/train/'
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

    def plot_relationship_stats(self):
        f = self._family_df.family.unique()
        number_of_relations = []
        for family in f:
            number_of_relations.append(self._family_df.loc[self._family_df.family == family].values.shape[0])
        plt.hist(number_of_relations, bins=100)  # most families have 10 relations, there are entries for 70+ relations
        print("Checking number of entries: ", len(self._train_relationships_df), np.cumsum(number_of_relations)[-1])
        plt.show()

    def show_relationship_example(self):
        f = np.random.randint(0,self._family_df.family.nunique())
        family_name = self._family_df.family.iloc[f]
        # get a dataframe for the sample family relationships
        temp = self._family_df.loc[self._family_df.family == family_name]
        p1_names = temp.p1.unique().tolist()
        p2_names = temp.p2.unique().tolist()
        total_names = list(set(p1_names + p2_names))
        print("Sampling for family: ", family_name)
        print(temp)

        for name in total_names:
            person_path = os.path.join(self._root_dir, family_name, name)
            try:
                images = os.listdir(person_path)
                img_path = os.path.join(person_path, np.random.choice(images))
                img = plt.imread(img_path)
                plt.figure()
                plt.imshow(img)
            except FileNotFoundError:
                print("Data is missing for this person: ", person_path)
        plt.show()

    def get_family_df(self):
        return self._family_df.copy()


def main():
    fd = FacialData()
    fd.show_relationship_example()
    # fd.print_info()
    # fd.plot_relationship_stats()


if __name__ == '__main__':
    main()