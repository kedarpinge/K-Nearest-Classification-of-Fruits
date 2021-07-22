import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KNeighborsClassifier

# Import File
fruits = pd.read_table('fruit_data_with_colors.txt')
print(fruits.head())

# Create Dictionary to View labels more accurately
lookup_fruit_name = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique()))
print(lookup_fruit_name)

# Assign Features and Labels
X = fruits[['mass', 'width', 'height', 'color_score']]
y = fruits['fruit_label']

# Split Training data into Train and Split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Feature Pair Plot

# 2D
cmap = cm.get_cmap('gnuplot')
scatter = pd.plotting.scatter_matrix(X_train, c=y_train, marker='o', s=40, hist_kwds={'bins': 15}, figsize=(12, 12),
                                     cmap=cmap)



# 3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train['width'], X_train['height'], X_train['color_score'], c=y_train, marker='o', s=100)
ax.set_xlabel('width')
ax.set_ylabel('height')
ax.set_zlabel('color_score')


# Questions to ask when using KNN

# Distance Metric (e.g. Euclidean : Minkowski with p = 2)
# How many Neighbours (e.g. k = 5)
# Optional Weighting Function (e.g. Ignore or more weightage on closer points)
# How to aggregate classes from the NN point (e.g. Majority in Classes)

# Create Classifier Object
knn = KNeighborsClassifier(n_neighbors=5)

# Train the Classifier
knn.fit(X_train, y_train)

# Estimate Accuracy of Classifier on Future Data
print(knn.score(X_test, y_test))

# Predict Individual, New Unseen Objects by providing features
fruit_prediction = knn.predict([[20, 4.3, 5.5, 0.4]])
print(lookup_fruit_name[fruit_prediction[0]])

# PLOT Sensitivity of KNN Classifier Accuracy to 'k' Parameter
k_range = range(1,20)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))

plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([0,5,10,15,20]);

# PLOT Sensitivity of KNN Classifier Accuracy to Train/test Split Proportion
t = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

knn = KNeighborsClassifier(n_neighbors = 5)

plt.figure()

for s in t:

    scores = []
    for i in range(1,1000):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1-s)
        knn.fit(X_train, y_train)
        scores.append(knn.score(X_test, y_test))
    plt.plot(s, np.mean(scores), 'bo')

plt.xlabel('Training set proportion (%)')
plt.ylabel('accuracy');

plt.show()

