import plotly.graph_objects as go
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# CONTOUR PLOT OF CONFIDENCE OF 'SUCCESS' LABEL GIVEN TEST DATA

#1 GET DATA AND CLASSIFIER
# Load and split data
X, y = make_moons(noise=0.3, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y.astype(str), test_size=0.25, random_state=0)

# Create classifier, run predictions on grid
clf = KNeighborsClassifier(15, weights='uniform')
clf.fit(X_train, y_train)


#2 PLOT SCATTER DIAGRAM OF FEATURES and their LABELS
trace_specs = [
    [X_train, y_train, '0', 'Train', 'square'],
    [X_train, y_train, '1', 'Train', 'circle'],
    [X_test, y_test, '0', 'Test', 'square-dot'],
    [X_test, y_test, '1', 'Test', 'circle-dot']
]

fig = go.Figure(data=[
    go.Scatter(
        x=X[y==label, 0], y=X[y==label, 1],
        name=f'{split} Split, Label {label}',
        mode='markers', marker_symbol=marker
    )
    for X, y, label, split, marker in trace_specs
])
fig.update_traces(
    marker_size=12, marker_line_width=1.5,
    marker_color="lightyellow"
)


#3 MAKE PREDICTION PROBABILITY BASED ON TRAIN DATA
# Create a mesh grid on which we will run our model
mesh_size, margin = .02, 0.25
x_min, x_max = X_train[:, 0].min() - margin, X_train[:, 0].max() + margin
y_min, y_max = X_train[:, 1].min() - margin, X_train[:, 1].max() + margin
xrange = np.arange(x_min, x_max, mesh_size)
yrange = np.arange(y_min, y_max, mesh_size)
xx, yy = np.meshgrid(xrange, yrange)

prediction = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
prediction = prediction.reshape(xx.shape)

# Equivalent of single points, not used later on. 
y_score = clf.predict_proba(X_test)[:, 1]


#4 PLOT CONTOUR FIGURE
fig.add_trace(
    go.Contour(
        x=xrange,
        y=yrange,
        z=prediction,
        showscale=False,
        colorscale='RdBu',
        opacity=0.4,
        name='Score',
        hoverinfo='skip'
    )
)

fig.show()



