from TREE.tree import *
import operator

test_point = ['vhigh', 'low', '3', '4', 'med', 'med']

def classify(datapoint, tree):
  # If the tree is a leaf, return the label with the highest count: this is the classification of that leaf
  if isinstance(tree, Leaf):
    return max(tree.labels.items(), key=operator.itemgetter(1))[0]
  value = datapoint[tree.feature]
  for branch in tree.branches:
    if branch.value == value:
      return classify(datapoint, branch)

print(classify(test_point,tree))
