

import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('exotrain.csv')
dt = pd.read_csv('exotest.csv')

# =============================================================================
# null values and correlation using heatmap
# =============================================================================
import seaborn as sns
sns.heatmap(df.iloc[:,1:].isnull())
sns.heatmap(df.corr(), fmt = ".2g", cmap = "ylgnbu", linewidths =".1")


# =============================================================================
# GAUSSIAN HISTOGRAM
# =============================================================================
labels_1=[100,200,300]
for i in labels_1:
    plt.figure(figsize=(3,3))
    plt.hist(df.iloc[i,:], bins=200)
    plt.title("gaussian histogram")
    plt.xlabel("flux values")
    plt.show()
    
labels_1=[3,5,4]
for i in labels_1:
    plt.figure(figsize=(3,3))
    plt.hist(df.iloc[i,:], bins=200)
    plt.title("gaussian histogram")
    plt.xlabel("flux values")
    plt.show()
    
    
# =============================================================================
# BOX PLOTS OF FLUX TO FIND OUTLIERS
# =============================================================================

fig, axes = plt.subplots(1,5,figsize =(15,6), sharey = True)
fig.suptitle('distribution of flux')

sns.boxplot(ax= axes[0], data = df, x = 'LABEL', y = 'FLUX.1', palette="Set2")
sns.boxplot(ax= axes[1], data = df, x = 'LABEL', y = 'FLUX.2', palette="Set2")
sns.boxplot(ax= axes[2], data = df, x = 'LABEL', y = 'FLUX.3', palette="Set2")
sns.boxplot(ax= axes[3], data = df, x = 'LABEL', y = 'FLUX.4', palette="Set2")
sns.boxplot(ax= axes[4], data = df, x = 'LABEL', y = 'FLUX.5', palette="Set2")


# =============================================================================
# DROP THE OUTLIERS
# =============================================================================

df.drop(df[df['FLUX.1']>250000].index, axis=0, inplace = True)

# =============================================================================
# BOX PLOTS OF FLUX WITHOUT OUTLIERS
# =============================================================================

fig, axes = plt.subplots(1,5,figsize =(15,6), sharey = True)
fig.suptitle('distribution of flux')

sns.boxplot(ax= axes[0], data = df, x = 'LABEL', y = 'FLUX.1', palette="Set2")
sns.boxplot(ax= axes[1], data = df, x = 'LABEL', y = 'FLUX.2', palette="Set2")
sns.boxplot(ax= axes[2], data = df, x = 'LABEL', y = 'FLUX.3', palette="Set2")
sns.boxplot(ax= axes[3], data = df, x = 'LABEL', y = 'FLUX.4', palette="Set2")
sns.boxplot(ax= axes[4], data = df, x = 'LABEL', y = 'FLUX.5', palette="Set2")

# =============================================================================
# TRAIN TEST DATA 
# =============================================================================
X_train = df.iloc[:,1:].values
y_train = df.iloc[:,0].values
X_test = dt.iloc[:,1:].values
y_test = dt.iloc[:,0].values

# =============================================================================
# NORMALISE
# =============================================================================
X_train = normalize(X_train)
X_test = normalize(X_test)

# =============================================================================
# MAPPING
# =============================================================================

label = {2:1, 1:0}
y_train = [label[i] for i in y_train]
# print(y_train)

label = {2:1, 1:0}
y_test = [label[i] for i in y_test]
# print(y_test)

# =============================================================================
# USING DECISION TREE REGRESSOR
# =============================================================================
from sklearn.tree import DecisionTreeRegressor
dectree = DecisionTreeRegressor(random_state = 0)

dectree.fit(X_train, y_train)
print(X_train.shape)

exo_predict = dectree.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,exo_predict,normalize = True)
print(accuracy)




