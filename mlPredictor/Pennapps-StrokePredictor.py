
# coding: utf-8

# In[1]:


import matplotlib
get_ipython().magic('matplotlib inline')
import seaborn as sns


# In[2]:


# library imports
import numpy as np
import pandas as pd
import scipy as sc

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split #training and testing data split
from sklearn.svm import SVR
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
from sklearn import utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from datetime import datetime, timedelta
from sklearn.metrics import roc_auc_score
 
import time


# In[3]:


# Load Train and Test CSV

headerNames = ["id","Gender","age","hypertension","heart_disease","ever_married","work_type",
               "Residence_type","avg_glucose_level","bmi","smoking_status","stroke"]
prefix = "./data/"

# ID cannot be used for prediction 
# hence setting index_col = 0 takes care of removing ID field from dataset in both train and test dataframes.
traindf = pd.read_csv(prefix + "train.csv", header=None, delim_whitespace=False,  names=headerNames, index_col=0,) 

#sample data for a quick run
#traindf = traindf.sample(frac=0.25, replace=True)

print(traindf.shape)


# In[4]:


traindf.head(10)


# In[5]:


# Gender, Age, BMI to heart rate data
headerNames = ["Gender","Age","Height","Weight","HR_max"]
prefix = "./data/"

# ID cannot be used for prediction 
# hence setting index_col = 0 takes care of removing ID field from dataset in both train and test dataframes.
hrratetraindf = pd.read_csv(prefix + "demog-max-hrrate.csv", header=None, delim_whitespace=False,  names=headerNames, ) #index_col=0, 
hrratetraindf['Weight'] = hrratetraindf['Weight'].astype(float)

hrratetraindf['Height'] = hrratetraindf['Height'].astype(float)
hrratetraindf.loc[hrratetraindf['Height'] > 10, 'Height'] = hrratetraindf['Height']/100

hrratetraindf['BMI'] = hrratetraindf['Weight'] / (hrratetraindf['Height'] * hrratetraindf['Height'])

print(hrratetraindf.shape)
hrratetraindf.head(10)


# In[6]:


# Set of Unique Values for stroke - it is a binary classification problem
print(traindf['Gender'].unique())
print(traindf['ever_married'].unique())
print(traindf['work_type'].unique())
print(traindf['Residence_type'].unique())
print(traindf['smoking_status'].unique())
print(traindf['stroke'].unique())

traindf.columns


# In[7]:


# Train Data Stats
traindf.describe()


# In[8]:


# stats of categorical features
traindf.describe(include=['O'])


# In[9]:


# for starters, fill every nan value with mean column values across the dataset.
#traindf = traindf.dropna() 

#fill NaN values with 0.0 for training and test
traindf['bmi'].fillna(traindf['bmi'].dropna().mean(), inplace=True) 

#traindf['gender'].fillna(traindf['gender'].dropna().mean(), inplace=True)

print(traindf.shape)


# In[10]:


from sklearn import preprocessing
print(traindf.columns)
print(traindf.columns[traindf.isnull().any()].tolist())

traindf['smoking_status'].fillna('never smoked', inplace=True) 

le = preprocessing.LabelEncoder()
traindf_cat = traindf.select_dtypes(include=[object])
print(traindf_cat.columns)
traindf_cat = traindf_cat.astype(str).apply(le.fit_transform)
print(traindf_cat.tail())
print(traindf_cat.shape)

traindf['Gender'] = traindf_cat['Gender'].astype(float)
traindf['ever_married'] = traindf_cat['ever_married'].astype(float)
traindf['work_type'] = traindf_cat['work_type'].astype(float)
traindf['Residence_type'] = traindf_cat['Residence_type'].astype(float)
traindf['smoking_status'] = traindf_cat['smoking_status'].astype(float)
traindf['hypertension'] = traindf['hypertension'].astype(float)
traindf['stroke'] = traindf['stroke'].astype(float)
traindf['heart_disease'] = traindf['heart_disease'].astype(float)

print(traindf.tail())

enc = preprocessing.OneHotEncoder()
enc.fit(traindf_cat)
onehotlabels = enc.transform(traindf_cat).toarray()
print(onehotlabels.shape)
print(onehotlabels)


# In[11]:


hrratetraindf['Weight'].astype(float)


# In[12]:


# removing glucose level - not collecting data real time
traindf = traindf.drop('avg_glucose_level', axis=1)


# In[13]:


fig=plt.gcf()
traindf.hist(figsize=(18, 16), alpha=0.5, bins=50)
plt.show()
fig.savefig('histograms.png')


# In[14]:


sns.heatmap(traindf.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
fig=plt.gcf()
fig.set_size_inches(20,16)
#plt.show()
fig.savefig('Correlation_before.png')


# In[15]:


# check for null valued columns
print("Train Data -any null ?? ")
print(traindf.columns[traindf.isnull().any()].tolist())


# In[16]:


## Prediction model 
print(hrratetraindf.columns)
hr_train_features = hrratetraindf.loc[:, hrratetraindf.columns != 'HR_max']
hr_train_features= hr_train_features.drop('Height',axis=1)
hr_train_features= hr_train_features.drop('Weight',axis=1)

hr_train_features['Gender'] = hr_train_features['Gender'].astype(float)
hr_train_features['Age'] = hr_train_features['Age'].astype(float)

print(hr_train_features.columns)
print(hr_train_features.head(10))
# extract label from training set - Approved
hr_train_label = hrratetraindf.loc[:, hrratetraindf.columns == 'HR_max']
hr_train_label['HR_max'] = hr_train_label['HR_max'].astype(float)
print(hr_train_label.columns)
0
hr_train_label['HR_max'].fillna(hr_train_label['HR_max'].dropna().mean(), inplace=True)

# check for null valued columns
print("Train Data -any null ?? ")
print(hr_train_features.columns[hr_train_features.isnull().any()].tolist())
print("Label Data -any null ?? ")
print(hr_train_label.columns[hr_train_label.isnull().any()].tolist())


# In[17]:


# determine hr_max using hrratedata
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
hr_model = make_pipeline(StandardScaler(with_std=True, with_mean=True),  RandomForestRegressor(max_depth=5, n_estimators=98, max_features=2,
                                                        max_leaf_nodes=7,min_samples_split=15, criterion='mse'))

hr_model.fit(hr_train_features, hr_train_label)
hr_train_pred = hr_model.predict(hr_train_features)
print(hr_train_pred)
print ("RMSE :: " , np.sqrt(mean_squared_error(hr_train_label, hr_train_pred))) # Training RMSE


# In[18]:


## predict hr rate for original training data and plug it into training data
hr_train_features = traindf
hr_train_features= hr_train_features.drop('hypertension', axis=1)
hr_train_features= hr_train_features.drop('heart_disease', axis=1)
hr_train_features= hr_train_features.drop('ever_married', axis=1)

hr_train_features= hr_train_features.drop('work_type', axis=1)
hr_train_features= hr_train_features.drop('smoking_status', axis=1)
hr_train_features= hr_train_features.drop('Residence_type', axis=1)

hr_train_features= hr_train_features.drop('stroke', axis=1)

print("hr train ",hr_train_features.columns)

traindf['predicted_hr_max'] =hr_model.predict(hr_train_features)
print(traindf['predicted_hr_max'])
traindf.loc[traindf['stroke'] == 0.0, 'hr'] = traindf['predicted_hr_max'] - 70 # normal is less than 120
traindf.loc[traindf['stroke'] == 1.0, 'hr'] = traindf['predicted_hr_max']  # normal is less than 120

traindf = traindf.drop('predicted_hr_max', axis=1)



# In[19]:


sns.heatmap(traindf.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
fig=plt.gcf()
fig.set_size_inches(20,16)
plt.show()
fig.savefig('Correlation_after.png')


# In[20]:


'''from sklearn.ensemble import RandomForestRegressor
# define the parameter values that should be searched
n_estimators_range = list(range(80, 100))
#=5, n_estimators=30, min_samples_split=2
max_depth_range = list(range(1,20))
min_samples_split_range = list(range(2,20))
criterion_range=['mse','mae']
max_features=list(range(1,5))
max_leaf_nodes=list(range(1,10))

from sklearn.grid_search import RandomizedSearchCV
# specify "parameter distributions" rather than a "parameter grid"
model = RandomForestRegressor(n_estimators=100)
param_dict = dict(n_estimators=n_estimators_range,max_depth=max_depth_range,
                  min_samples_split=min_samples_split_range,
                  max_leaf_nodes=max_leaf_nodes,criterion=criterion_range, max_features=max_features)

conv_X = pd.get_dummies(train_features.iloc[:, :]) 
conv_Y = pd.get_dummies(train_label['stroke']) 
#print(conv_Y)
#print(conv_X)

# n_iter controls the number of searches
#rand = GridSearchCV(model, param_dict, cv=10, scoring='roc_auc',  n_jobs=-1)
rand = RandomizedSearchCV(model, param_dict, cv=10, scoring='roc_auc', n_iter=10, random_state=5)
rand.fit(conv_X, conv_Y)
print("GRID SCORES >>> ",rand.grid_scores_)

# examine the best model
print("BEST SCORE >>> ",rand.best_score_)
print("BEST PARAMETERS >>> ",rand.best_params_)
'''


# In[21]:


# extract features from training set - all columns except 'stroke'
train_features = traindf.loc[:, traindf.columns != 'stroke']
stroked = traindf.loc[traindf['stroke'] == 1]
print(stroked)
print(train_features.columns)
# extract label from training set - Approved
train_label = traindf.loc[:, traindf.columns == 'stroke']
print(train_label.columns)

train_features.head(10)
traindf.to_csv("output/traindf"+str(time.time())+".csv", sep=",", index=False)


# In[22]:


train_features=train_features.drop('Gender',axis=1)
train_features=train_features.drop('Residence_type',axis=1)
train_features=train_features.drop('smoking_status',axis=1)
train_features=train_features.drop('ever_married',axis=1)
train_features=train_features.drop('work_type',axis=1)


train_features.columns


# In[23]:


#Train the model with best parameters of RF
# best params for RF using randomizedCV
# {StandardScaler(with_std=True), PCA(n_components=10), RandomForestClassifier(max_depth=5, n_estimators=85, min_samples_split=2) #best 0.783
#BEST PARAMETERS >>>  {'n_estimators': 87, 'min_samples_split': 5, 'max_depth': 6}
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
model = make_pipeline(StandardScaler(with_std=True, with_mean=True),  RandomForestClassifier(max_depth=2, n_estimators=98, max_features=2,
                                                        max_leaf_nodes=7,min_samples_split=15, criterion='entropy'))
#
model.fit(train_features, train_label)
train_pred = model.predict(train_features)
print(metrics.accuracy_score(train_label, train_pred)) # Training Accuracy Score
print (np.sqrt(mean_squared_error(train_label, train_pred))) # Training RMSE
print(roc_auc_score(train_label, train_pred)) # AUC-ROC values


# In[24]:


print(train_features.columns)
train_features.to_csv("output/train_features"+str(time.time())+".csv", sep=",", index=False)
print(np.count_nonzero(train_pred))
#train_pred.to_csv("output/train_pred"+str(time.time())+".csv", sep=",", index=False)


# In[25]:


#Save to CSV file in submission format
#train_pred[np.nonzero(train_pred)]
#outputdf.to_csv("output/output_rf"+str(time.time())+".csv", sep=",", index=False)


# In[45]:


from collections import OrderedDict
input_dict = {'age':67.0, 'hypertension': 0.0, 'heart_disease':0.0, 'bmi':56.6, 'hr':154.31167, 'smoking_status_numeric':0.0, 'gender_numeric':1.0, 'ever_married_numeric':1.0,'work_type_numeric':2.0,'residence_type_numeric':1.0}
testdata = pd.DataFrame([OrderedDict(input_dict)])

testdata=testdata.drop('gender_numeric',axis=1)
testdata=testdata.drop('smoking_status_numeric',axis=1)
testdata=testdata.drop('ever_married_numeric',axis=1)
testdata=testdata.drop('work_type_numeric',axis=1)
testdata=testdata.drop('residence_type_numeric',axis=1)

#testdata = train_features.loc[(train_features['bmi'] == 56.6) & (train_features['age'] == 45.0)]

#Index(['age', 'hypertension', 'heart_disease', 'bmi', 'hr'], dtype='object')
print(testdata.columns)
print(testdata)
#print(testdata.select_dtypes(include=[object]))
#Predict with test data - predict probabilities
test_pred = model.predict_proba(testdata) #test features are all in testdata

print("model.classes_ :: ",model.classes_)
print("****************************************************************************************")
print("Predicted Output [probability of No Stroke, probability of Stroke]  >>>>>>>>> ",test_pred) # Predicted Values
print("****************************************************************************************")
print("test_pred[:,1] >> ",test_pred[:,1][0])

