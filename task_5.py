#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# loading dataset
df = pd.read_csv('dataset/heart.csv')

# exploring data
print(df.head())
print(df.info())
print(df['target'].value_counts())

# feature and target split
x = df.drop('target', axis=1)
y = df['target']

# splitting the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

# train decision tree classifier
dtree = DecisionTreeClassifier(random_state = 42)
dtree.fit(x_train, y_train)
y_pred_dt = dtree.predict(x_test)

# evaluate decision tree
print('Decision Tree Accuracy: ', accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

# visualize decision tree
plt.figure(figsize = (16, 8))
plot_tree(dtree, feature_names = x.columns, class_names = ['No Disease', 'Disease'], filled = True, max_depth = 3)
plt.title('Decision Tree Visualization (depth = 3)')
plt.show()

# analyze overfitting by varying max_depth
train_scores, test_scores = [], []
for i in range(1, 15):
    temp_tree = DecisionTreeClassifier(max_depth = i, random_state = 42)
    temp_tree.fit(x_train, y_train)
    train_scores.append(temp_tree.score(x_train, y_train))
    test_scores.append(temp_tree.score(x_test, y_test))

plt.plot(range(1, 15), train_scores, label = 'Train')
plt.plot(range(1, 15), test_scores, label = 'Test')
plt.xlabel('Tree Depth')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Decision Tree Overfitting Analysis')
plt.show()

# train random forest classifier
rforest = RandomForestClassifier(n_estimators = 100, random_state = 42)
rforest.fit(x_train, y_train)
y_pred_rf = rforest.predict(x_test)

# evaluate random forest
print('Random Forest Accuracy: ', accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# compare with cross_validation
dtree_cv = cross_val_score(dtree, x, y, cv = 5).mean()
rforest_cv = cross_val_score(rforest, x, y, cv = 5).mean()
print('Decision Tree CV Accuracy:  {dtree_cv:.3f}')
print('Random Forest CV Accuracy: {rforest_csv:.3f}')

# feature importance
importances = rforest.feature_importances_
feat_imp = pd.Series(importances, index=x.columns).sort_values(ascending = False)
sns.barplot(x = feat_imp, y = feat_imp.index)
plt.title('Feature Importance (Random Forest)')
plt.show()