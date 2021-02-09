# =============================================================================
# OUTLIERS - FINDING AND REMOVING
# =============================================================================
mp.figure(figsize=(7,5))
sns.boxplot(x= 'Survived', y='Age', hue ='Sex', data = train)
test.plot(kind='box', figsize=(6,5))

# USING CLIP TO REMOVE OUTLIERS
test[cols]= test[cols].clip(lower= test[cols].quantile(0.15), upper= test[cols].quantile(0.85), axis=1)
test.drop(columns = ['Parch'], axis =1, inplace = True)

# ANOTHER WAY
print('Dropping Outliers')
train_data.drop(train_data[train_data['FLUX.1']>250000].index, axis=0, inplace=True)

# =============================================================================
# USING PLOTS TO VISULAISE DATA AND PREPROCESSING
# =============================================================================

plt.figure(figsize=(4,8))
colors = ["0", "1"]
sns.countplot('LABEL', data=train_data, palette = "Set2")
plt.title('Class Distributions \n (0: Not Exoplanet || 1: Exoplanet)', fontsize=14)

# PLOTS TO FIND MISSING VALUES 
sns.heatmap(train_data.isnull())

# =============================================================================
# USING PANDAS - NULL VALUES, DROP COLUMNS
# =============================================================================
train.isnull().sum()
train['Age'].fillna(train['Age'].median(), inplace=True)
train.isnull().sum()
train['Survived'].value_counts()

train.drop(columns = ['Name','Ticket','Cabin'], inplace = True)
ytrain = train['Survived'].values.reshape(-1,1)

new = ann
new.replace('?', -9999, inplace = True)
# for i in new:
#     if new[i]
print(new['formability'].value_counts())

# =============================================================================
# CORRELATION TO UNDESTAND RELATIONSHIP BETWEEN FEATURES
# =============================================================================
plt.figure(figsize=(15,15))
sns.heatmap(train_data.corr())
plt.title('Correlation in the data')
plt.show()

# =============================================================================
# USING PREPROCESSING FROM SKLEARN
# =============================================================================
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
features =['Age', 'SibSp', 'Fare'] 
# independent variables
xtrain[features]=ss.fit_transform(xtrain[features])
# xtest[features]=ss.fit_transform(xtest[features])

#Feature scaling
std_scaler = StandardScaler()
x_train = scaled = std_scaler.fit_transform(x_train)
x_test = std_scaler.fit_transform(x_test)

# NORMALISE
x_train = normalized = normalize(x_train)
x_test = normalize(x_test)

# ANOTHER WAY TO NORMALISE
X = X/255.0
X = np.array(X)
y = np.array(y)


# =============================================================================
# FILE PATH AND FILE I/O OPERATIONS
# =============================================================================

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

datadir = 'D:/PetImages'
categories =['dog','cat']
for category in categories:
    path = os.path.join(datadir, category)
    for img in os.listdir(path):
#         here color is not an important factor spo we usegrayscale
        img_array = cv2.imread(os.path.join(path, img),cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array, cmap='gray')
        plt.show()
        print('img_array:\n',img_array)
        print('shape is:\n', img_array.shape)
        break
    break
    

with open("dataset.csv") as file:
     a = file.read().replace(";", ",")       
with open("dataset.csv",'w') as file:
     file.write(a)   

x=''
c=0
with open("dataset.csv") as file:
    for i in file:
        if c == 0:
            a =1
        else:
            x = x+'\n'+i
        c=c+1
with open('abc.csv','w') as file:
    file.write(x)     
        
# =============================================================================
# MAPPING
# =============================================================================
categ = {2: 1,1: 0}
train_data.LABEL = [categ[item] for item in train_data.LABEL]
test_data.LABEL = [categ[item] for item in test_data.LABEL]

# WORKING

month_map={'jan':1, 'feb':2, 'mar':3, 'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
day_map={'mon':2, 'tue':3, 'wed':4, 'thu':5, 'fri':6, 'sat':7, 'sun':1}
X.month.replace(to_replace = month_map, inplace = True)

# USED EARLIER 

Xtemp = features
Xtemp.drop(columns = ['product', 'steel', 'shape'])

a= features['product'].map(product_map)
b=features['steel'].map(steel_map)
c=features['shape'].map(shape_map)
# print(a)
Xtemp['product']=a
Xtemp['steel']=b
Xtemp['shape']=c

# ANOTHER METHOD - BAD WAY
features['steel'].replace('A', 1, inplace=True)
features['steel'].replace('R', 2,inplace = True)
features['steel'].replace('K', 3, inplace=True)


# =============================================================================
# SHUFFLING DATA SO THAT MODEL AN LEARN BETTER 
# =============================================================================
import random
random.shuffle(training_data)

# =============================================================================
# SAVING DATA
# =============================================================================
import pickle
pickle_out = open('X.pickle', 'wb')
pickle.dump(X, pickle_out)
pickle_out.close()

# loading  data from pickle to variable
pickle_in = open('X.pickle', 'rb')
X = pickle.load(pickle_in)

# =============================================================================
# PROCESSING IMAGES USING CV2
# =============================================================================
img_size = 50 
new_array = cv2.resize(img_array, (img_size, img_size))
plt.imshow(new_array, cmap = 'gray')
plt.show()
print(new_array.shape)



