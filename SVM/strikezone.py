import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from svm_visualization import draw_boundary
from players import aaron_judge

fig, ax = plt.subplots()

def strike_zone(name):
  name.type = name.type.map({'S': 1, 'B':0})
  name = name.dropna(subset = ['plate_x', 'plate_z', 'type'])
  plt.scatter(name.plate_x, name.plate_z, c=name.type, cmap=plt.cm.coolwarm, alpha=0.25)

  train_data, validation_data = train_test_split(name, random_state=1)

  classifier = SVC(gamma = 0.05, C = 1000)
  classifier.fit(train_data[['plate_x', 'plate_z']], train_data['type'])
  draw_boundary(ax, classifier)
  print(classifier.score(validation_data[['plate_x', 'plate_z']], validation_data['type']))

  ax.set_ylim(-2, 6)
  ax.set_xlim(-3, 3)
  plt.show()


strike_zone(aaron_judge)
# strike_zone(jose_altuve)
# strike_zone(david_ortiz)



