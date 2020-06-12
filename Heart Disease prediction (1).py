#!/usr/bin/env python
# coding: utf-8

# # Heart Disease Prediction Model

# Heart disease describes a range of conditions that affect your heart.Diseases under the heart disease umbrella include blood vessel diseases, such as coronary artery disease, heart rhythm problems (arrhythmias) and heart defects you’re born with (congenital heart defects), among others.Heart disease is one of the biggest causes of morbidity and mortality among the population of the world.

# ## Overview of Dataset

# This database contains 76 attributes, but all published experiments refer to using a subset of 14 of them.<br>
# The dataset is a labelled dataset (has features determining the target) thus we'll use Supervised Learning models.<br>
# The target column is classifying the individuals have heart disease or not, thus we'll implement Classification models.<br>
# We used 4 classification model:<br>
#     &emsp;&emsp;&emsp;1.DecisionTree Classifier<br>
#     &emsp;&emsp;&emsp;2.RandomForest Classifier<br>
#     &emsp;&emsp;&emsp;3.Support Vector Machine<br>
#     &emsp;&emsp;&emsp;4.KnearestNeighbors <br>
#     &emsp;&emsp;&emsp;5.Logistic Regression<br>

# ### Import necessary packages

# In[1]:


#import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#import libraries for modelling
from sklearn.model_selection import train_test_split,cross_val_score,learning_curve,RandomizedSearchCV,KFold,LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix,auc,roc_curve,roc_auc_score,classification_report,make_scorer
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier,RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# DATA EXPLORATION

# In[3]:


#load dataset
data=pd.read_csv("../RN7/data/Heart Disease.csv")
original=pd.read_csv("../RN7/data/Heart Disease.csv")


# In[4]:


data.head()


# 1.age-age of individuals<br>
# 2.sex => gender of individuals.<br>
# &emsp; 1 for male and 0 for female<br>
# 3.cp=>Chest pain<br>
# &emsp;  1.value 1= 'typical angina'<br>
# &emsp;  2.value 2='atypical angina'<br>
# &emsp; 3.value 3= 'non-anginal'<br>
# &emsp;  4.value 4= 'asymptotic'<br>
# trestbps=>Resting blood pressure<br>
# chol=>serum cholestoral<br>
# fbs=>fasting Blood Sugar(1=true,0=false)<br>
# restecg=>resting electrocardiographic results<br>
# thalach=>maximum heart rate achieved <br>
# exang=>exercise included angina(1=yes,0=no)<br>
# oldpeak => ST depression induced by exercise relative to rest<br>
# slope=>the slope of the peak exercise ST segment<br>
# ca => number of major vessels (0-3) colored by flourosopy<br>
# thal: 3 = normal; 6 = fixed defect; 7 = reversable defect<br>

# In[5]:


data.thal.value_counts()


# In[6]:


data.info()


# In[7]:


data.describe()

This DATASET has 9 categorical variable
# 1.sex<br>
# 2.cp<br>
# 3.fbs<br>
# 4.restecg<br>
# 5.exang<br>
# 6.slope<br>
# 7.ca<br>
# 8.thal<br>
# 9.our target variable<br>

# In[ ]:





# In[8]:


#change column name
data.columns=['age','sex','chest_pain_type','resting_blood_pressure','cholestoral','fasting_blood_sugar',
              'rest_ecg','max_heart_rate achieved','exercise_induced_angina','st_deprssion','slope',
              'num_major_vessel','thalassemia','target']


# In[9]:


data.head()


# In[ ]:





# In[10]:


data.chest_pain_type.value_counts()


# Chest_pain_type:<br>
# &emsp;    (Since number start from 0)<br> 
# &emsp;    0 -- typical angina<br>
# &emsp;    1 --atypical angina<br>
# &emsp;    2 -- non-anginal<br>
# &emsp;    3 --asymptotic<br>
# fasting blood sugar:<br>
# &emsp;    1- blood sugar > 120<br>
# &emsp;    0- blood sugar <=120<br>
# rest_ecg:<br>
# &emsp;    0 --normal<br>
# &emsp;    1 -- ST-T wave abnormality<br>
# &emsp;    2 -- left ventricular hypertrophy<br>
# exercise_induced_angina:<br>
# &emsp;    0 --no<br>
# &emsp;    1 --yes<br>
# slope:<br>
# &emsp;    0 --upsloping<br>
# &emsp;    1 --flat<br>
# &emsp;    2 --downsloping<br>
# thalassemia:<br>
# &emsp;    1 --normal<br>
# &emsp;    2 --fixed defect<br>
# &emsp;    3 --reversable defect<br>
# target:<br>
# &emsp;    0 --absent<br>
# &emsp;    1,2,3 --present<br>

# CHANGE VALUES INTO CATEGORIES 

# In[11]:


#divide sex column into categories
data['sex'][data['sex']==0]='female'
data['sex'][data['sex']==1]='male'

#divide chest_pain_type column into categories
data['chest_pain_type'][data['chest_pain_type']==0]='typical angina'
data['chest_pain_type'][data['chest_pain_type']==1]='atypical angina'
data['chest_pain_type'][data['chest_pain_type']==2]='non-anginal'
data['chest_pain_type'][data['chest_pain_type']==3]='asymptotic'

#divide fasting_blood_sugar column into categories
data['fasting_blood_sugar'][data['fasting_blood_sugar']==1]='greater than 120 mg/ml'
data['fasting_blood_sugar'][data['fasting_blood_sugar']==0]='lower than 120 mg/ml'

#divide rest_ecg column into categories
data['rest_ecg'][data['rest_ecg']==0]='normal'
data['rest_ecg'][data['rest_ecg']==1]='ST-T wave abnormality'
data['rest_ecg'][data['rest_ecg']==2]='left ventricular hypertrophy'

#divide exercise_induced_angina column into categories
data['exercise_induced_angina'][data['exercise_induced_angina']==0]='no'
data['exercise_induced_angina'][data['exercise_induced_angina']==1]='yes'

#divide slope column into categories
data['slope'][data['slope']==0]='upsloping'
data['slope'][data['slope']==1]='flat'
data['slope'][data['slope']==2]='downsloping'


# In[12]:


data.head()


# In[13]:


data.info()


# In[14]:


data.isnull().sum()


# There are no missing values

# ANALYZE DATA BY VISUALISATION

# In[15]:


size=[data[data.target==0].count().target,data[data.target==1].count().target]
label=['absent','present']
fig,ax=plt.subplots(figsize=(8,5))
ax.pie(size,labels=label,autopct='%1.1f%%',shadow=True,)
ax.axis('equal')
plt.show()


# In this dataset 46% of the individuals don't have heart diesease while 54% have heart diesease

# In[16]:


plt.figure(figsize=(10,5))
sns.countplot(x='sex',data=data,hue='target',palette="GnBu")


# male are more likely to have heart diesease

# In[17]:


fig,ax=plt.subplots(1,2,figsize=(16,7))
label=['absent','present']
data[data['sex']=='male'].target.value_counts().plot.pie(explode=[0,0.10],autopct='%1.1f%%',ax=ax[0],shadow=True,labels=label)
data[data['sex']=='female'].target.value_counts().plot.pie(explode=[0,0.10],autopct='%1.1f%%',ax=ax[1],shadow=True,labels=['present','absent'])
plt.show()


# 45% of men and 75% of female have heart disease.<br>
# percentage of female patients is more than male patient in this dataset.<br>
# rate of female patient is way more than the female with no heart disease.<br>

# In[18]:


plt.figure(figsize=(10,5))
sns.countplot(x='chest_pain_type',data=data,hue='target',palette="mako_r")


# Indiviuals with non-anginal type chest pain is more in patient with heart disease category .So it can be indication for heart disease.<br>
# typical angina type chest pain is not that serious since there are more healthy individuals in this catgory as compare to patient

# In[19]:


plt.figure(figsize=(10,5))
sns.countplot(x='fasting_blood_sugar',data=data,hue='target',palette="mako_r")


# amount of fasting blood sugar is almost same in both category . so it can't be a good indicator of heart disease.<br>
# But Most of the heart disease patients have blood sugar less than 120.

# In[20]:


plt.figure(figsize=(10,5))
sns.countplot(x='rest_ecg',data=data,hue='target',palette="mako_r")


# In[21]:


plt.figure(figsize=(10,5))
sns.countplot(x='exercise_induced_angina',data=data,hue='target')


# exercise induced angina --amount of oxygen in the heart due to excercise that cause chest pain.<br>
# People without exercise induced angina is more in the category with disease.

# In[22]:


plt.figure(figsize=(10,5))
sns.countplot(x='slope',data=data,hue='target',palette="mako_r")


# people in downsloping category have high risk for heart disease.<br>

# In[23]:


plt.figure(figsize=(10,5))
sns.countplot(x='thalassemia',data=data,hue='target',palette="mako_r")


# Most of the people with heart disease have thal as 2.<br>

# In[24]:


plt.figure(figsize=(10,5))
sns.countplot(x='num_major_vessel',data=data,hue='target',palette="mako_r")


# Most of the people with heart disease have ca as 0.<br>

# In[25]:


sns.pairplot(data)


# In[26]:


plt.figure(figsize=(10,5))
sns.boxplot(x='target',y='age',data=data)


# median of a healthy person is more than  heart patient.<br>
# most of the heart patient lies between the age of 42 and 58 while there are also healthy person whose age lies btween those peak period.Hence age can't be a good indicator

# In[27]:


plt.figure(figsize=(10,5))
sns.boxplot(x='target',y='max_heart_rate achieved',data=data)


# Most of the patient have heart rate  between 150 and 170.<br>
# so heart rate above 150 can be considered as sign of danger.<br>
# But there are some cases where patient have heart rate less tan 110.<br>

# In[28]:


plt.figure(figsize=(10,5))
sns.boxplot(x='target',y='resting_blood_pressure',data=data)


# In[29]:


plt.figure(figsize=(10,5))
sns.boxplot(x='target',y='cholestoral',data=data)


# Most of the positive cases have cholestrol level between 200 to 270.<br>
# There are some outlier in positive case category which shows that high amount cholestoral affect heart .<br>
# cholestoral level betwen 250 and 500 considered to be high whil above 500 considered to be very high.<br>
# 

# In[30]:


sns.lmplot(x='age',y='cholestoral',data=data,hue='target',aspect=1.5)

cholestoral level slightly increases with increase in age of individuals with heart disease
# In[31]:


sns.lmplot(x='age',y='resting_blood_pressure',data=data,hue='target',aspect=1.5)


# In[32]:


sns.lmplot(x='age',y='max_heart_rate achieved',data=data,hue='target',aspect=1.5)

peak heart rate achieved during the age of 35 to 50 by individuals with heart disease 

# In[33]:


sns.lmplot(x='resting_blood_pressure',y='cholestoral',data=data,hue='target',aspect=1.5)


# In[34]:


sns.pairplot(data[['age','resting_blood_pressure','cholestoral','max_heart_rate achieved','st_deprssion','target']])


# There is no notable relationship between contionus features .<br>
# But we can see that 'st_depression' has right-skewed distribution so that we can apply log transformation.<br>

# In[35]:


plt.figure(figsize=(15,8))
corr=data.corr()
sns.heatmap(corr,annot=True)


# None of the features are highly corelated.So we don't need any kind of feature selection and feature extraction..<br>

# DETECTING OUTLIER
1.CHOLESTORAL
# In[36]:


plt.figure(figsize=(10,5))
sns.boxplot(x='cholestoral',data=data)


# In[37]:


data.cholestoral.describe()


# In[38]:


IQR=data.cholestoral.quantile(0.75)-data.cholestoral.quantile(0.25)
upper=data.cholestoral.quantile(0.75)+(IQR*1.5)
lower=data.cholestoral.quantile(0.25)-(IQR*1.5)
lower,IQR,upper


# In[39]:


IQR_x=data.cholestoral.quantile(0.75)-data.cholestoral.quantile(0.25)
upper_x=data.cholestoral.quantile(0.75)+(IQR*3)
lower_x=data.cholestoral.quantile(0.25)-(IQR*3)
lower_x,IQR_x,upper_x


# In[40]:


x=data[data.cholestoral>465]
y=data[data.cholestoral>369]
y

RESTING_BLOOD_PRESSURE
# In[41]:


plt.figure(figsize=(10,5))
sns.boxplot(x='resting_blood_pressure',data=data)


# In[42]:


data.resting_blood_pressure.describe()


# In[43]:


IQR=data.resting_blood_pressure.quantile(0.75)-data.resting_blood_pressure.quantile(0.25)
upper=data.resting_blood_pressure.quantile(0.75)+(IQR*1.5)
lower=data.resting_blood_pressure.quantile(0.25)-(IQR*1.5)
lower,IQR,upper


# In[44]:


y=data[data.resting_blood_pressure>170]
y


# In[45]:


IQR=data["max_heart_rate achieved"].quantile(0.75)-data["max_heart_rate achieved"].quantile(0.25)
upper=data["max_heart_rate achieved"].quantile(0.75)+(IQR*1.5)
lower=data["max_heart_rate achieved"].quantile(0.25)-(IQR*1.5)
lower,IQR,upper


# In[46]:


y=data[data["max_heart_rate achieved"]<85]
y


# DATA PREPARATION

# In[47]:


from sklearn.preprocessing import PowerTransformer
log=PowerTransformer()
log.fit(data[['st_deprssion']])
data['log_depression']=log.transform(data[['st_deprssion']])
data.drop('st_deprssion',inplace=True,axis=1)


# In[48]:


cnts_feature=['age','resting_blood_pressure','cholestoral','max_heart_rate achieved','log_depression']
cat_feature=[i for i in data.columns if i not in cnts_feature + ['target']]


# In[49]:


data=pd.get_dummies(data,columns=cat_feature)


# In[50]:


X=data.drop('target',axis=1)
y=data['target']


# In[242]:


#split data into train and test set
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)


# NORMALISATION

# In[243]:


scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


# MODEL

# model preparation

# In[244]:


#instantiate different model for tuning
tree=DecisionTreeClassifier(random_state=42)
forest=RandomForestClassifier()
lr=LogisticRegression()
svc=SVC()


# In[245]:


#initialise hyperparameter for tuning
random_forest_param={'n_estimators':[int(x) for x in np.linspace(start=100,stop=2000,num=10)],
                     'criterion':['gini','entropy'],
                     'max_features':['auto','sqrt',None],
                     'max_depth':[int(x) for x in np.linspace(start=5,stop=50,num=15)]+[None],
                     'min_samples_split':[2,5,10],
                     'min_samples_leaf':[1,2,4]}
decision_tree_param={'criterion':['gini','entropy'],
                     'min_samples_leaf':[1,2,3,4]}
svm_param = {'C':[0.001, 0.01, 0.1, 1],
             'gamma':[0.00001, 0.0001, 0.001,0.005, 0.01,0.05, 0.1],
             'kernel':["linear","rbf"]}


# In[246]:


#function for RandomizedSearchCV
def search(estimator,parameter):
    tune_model=RandomizedSearchCV(estimator=estimator,
                                      param_distributions=parameter,
                                      n_iter=100,cv=5,verbose=2,
                                      random_state=42,n_jobs=-1,refit=True)
    tune_model.fit(X_train,y_train)
    return tune_model


# In[247]:


#function to evaluate the classification model
def metrics(test,prediction):
    return confusion_matrix(test,prediction,[0,1]),classification_report(test,prediction),roc_auc_score(test,prediction)


# In[248]:


#function to draw roc_auc_curve
def curve(test,prediction):
    fpr, tpr, thresholds = roc_curve(test, prediction)
    auc_ = auc(fpr, tpr)
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label=' (area = {:.3f})'.format(auc_))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()


# ### 1.DecisionTree Classifier

# In[175]:


#search for best hyperparameter
decision_tree=search(tree,decision_tree_param)


# In[176]:


#predict on unseen test data 
y_pred=decision_tree.predict(X_test)


# In[177]:


#accuracy of model using classsification metrics
matrix,report,score=metrics(y_test,y_pred)


# In[178]:


#confusion matrix of Random forest Classifier
print("confusion matrix is given below:")
sns.heatmap(matrix, annot=True, fmt='.2f',xticklabels = ["healthy","patient"] , yticklabels = ["healthy","patient"])
plt.ylabel('True class')
plt.xlabel('Predicted class')


# In[179]:


tn, fp, fn, tp = matrix.ravel()
print(" confusion matrix shows that  we correctly classified {} and misclassified {} cases out of {} ".format(tn+tp,fp+fn,tn+tp+fp+fn))


# In[180]:


print("value of false negative {}".format(fn))


# In[181]:


print("We got {} accuracy on test data".format(accuracy_score(y_test,y_pred)))


# In[182]:


test_acc1=accuracy_score(y_test,y_pred)
train_acc1=decision_tree.best_score_


# In[183]:


print("classification report is given below:")
print(report)


# In[184]:


print("roc-auc-score of model is {}".format(roc_auc_score(y_test,y_pred)))


# In[185]:


#ROC_AUC_CURVE IS GIVEN BELOW
curve(y_test,y_pred)


# ### 2.RandomForestClassifier

# In[249]:


#search for best hyperparameter
random_forest=search(forest,random_forest_param)


# In[250]:


#predict on unseen test data 
y_pred=random_forest.predict(X_test)


# In[251]:


#accuracy of model using classsification metrics
matrix,report,score=metrics(y_test,y_pred)


# In[252]:


#confusion matrix of Random forest Classifier
print("confusion matrix is given below:")
sns.heatmap(matrix, annot=True, fmt='.2f',xticklabels = ["healthy","patient"] , yticklabels = ["healthy","patient"])
plt.ylabel('True class')
plt.xlabel('Predicted class')


# In[253]:


tn, fp, fn, tp = matrix.ravel()
print(" confusion matrix shows that  we correctly classified {} and misclassified {} cases out of {} ".format(tn+tp,fp+fn,tn+tp+fp+fn))


# In[254]:


print("value of false negative {}".format(fn))


# one of the most important metrics in medical cases for classification problm is false negative which represtent that observation is positive but prediction is negative .In our dataset value of False negative is quite low which is a good indicator of accuracy of our model

# In[255]:


print("We got {} accuracy on test data".format(accuracy_score(y_test,y_pred)))


# In[256]:


test_acc2=accuracy_score(y_test,y_pred)
train_acc2=random_forest.best_score_


# In[257]:


print("classification report is given below:")
print(report)


# for class 0:<br>
# &emsp;    1.precision is 87% which means that 87% of label 0 is corectly classified among total number of predicted negative                 example<br>
# &emsp;    2.recall is 77% which means that 77% of label 0 is corectly classified among total number of negative example<br>
# for class 1:<br>
# &emsp;     1.precision is 81% which means that 81% of label 1 is corectly classified among total number of predicted positive           example<br>
# &emsp;    2.recall is 89% which means that 89% of label 1 is corectly classified among total number of positive example<br>

# In[197]:


print("roc-auc-score of model is {}".format(roc_auc_score(y_test,y_pred)))


# In[198]:


#ROC_AUC_CURVE IS GIVEN BELOW
curve(y_test,y_pred)


# #### SVM Classifier

# In[192]:



svm=RandomizedSearchCV(svc,svm_param,cv=5,n_jobs=-1,refit=True,random_state=42)
svm.fit(X_train,y_train)


# In[193]:


svm.best_params_


# In[194]:


#predict on unseen test data 
y_pred=svm.predict(X_test)


# In[195]:


#accuracy of model using classsification metrics
matrix,report,score=metrics(y_test,y_pred)


# In[196]:


#confusion matrix of Random forest Classifier
print("confusion matrix is given below:")
sns.heatmap(matrix, annot=True, fmt='.2f',xticklabels = ["healthy","patient"] , yticklabels = ["healthy","patient"])
plt.ylabel('True class')
plt.xlabel('Predicted class')


# In[197]:


tn, fp, fn, tp = matrix.ravel()
print(" confusion matrix shows that  we correctly classified {} and misclassified {} cases out of {} ".format(tn+tp,fp+fn,tn+tp+fp+fn))


# In[198]:


print("value of false negative {}".format(fn))


# In[199]:


print("We got {} accuracy on test data".format(accuracy_score(y_test,y_pred)))


# In[200]:


test_acc3=accuracy_score(y_test,y_pred)
train_acc3=svm.best_score_


# In[201]:


print("classification report is given below:")
print(report)


# In[202]:


print("roc-auc-score of model is {}".format(roc_auc_score(y_test,y_pred)))


# In[203]:


#ROC_AUC_CURVE IS GIVEN BELOW
curve(y_test,y_pred)


# 3.KNearestNeighbors Classifier

# In[204]:


#try KnearestNeighbor Classifier for different value of n_neighbors
n=[x for x in list(range(1,50))]
train_scores=[]
test_scores=[]
for k in n:
    k_value = k
    knn=KNeighborsClassifier(n_neighbors = k_value)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    test_scores.append(accuracy_score(y_test,y_pred))
    train_scores.append(accuracy_score(y_train,knn.predict(X_train)))    


# In[205]:


from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(n, train_scores, 'b', label='Train AUC')
line2, = plt.plot(n, test_scores, 'r', label='Test AUC')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.show()


# from above figure we found that for n_neighbors=8 the model predict perfectly.<br>
# But with increase in value of n_neighbors model started overfitting.<br>

# In[206]:


Knearest=KNeighborsClassifier(n_neighbors=8)
Knearest.fit(X_train,y_train)


# In[207]:


#predict on unseen test data 
y_pred=Knearest.predict(X_test)


# In[208]:


#accuracy of model using classsification metrics
matrix,report,score=metrics(y_test,y_pred)


# In[209]:


#confusion matrix of Random forest Classifier
print("confusion matrix is given below:")
sns.heatmap(matrix, annot=True, fmt='.2f',xticklabels = ["0","1"] , yticklabels = ["0","1"])
plt.ylabel('True class')
plt.xlabel('Predicted class')


# In[210]:


tn, fp, fn, tp = matrix.ravel()
print(" confusion matrix shows that  we correctly classified {} and misclassified {} cases out of {} ".format(tn+tp,fp+fn,tn+tp+fp+fn))


# In[211]:


print("value of false negative {}".format(fn))


# one of the most important metrics in medical cases for classification problm is false negative which represtent that observation is positive but prediction is negative .In our dataset value of False negative is quite low which is a good indicator of accuracy of our model

# In[212]:


print("We got {} accuracy on test data".format(accuracy_score(y_test,y_pred)))


# In[213]:


test_acc4=accuracy_score(y_test,y_pred)
train_acc4=accuracy_score(y_train,Knearest.predict(X_train))


# In[214]:


accuracy_score(y_train,Knearest.predict(X_train))


# In[215]:


print("classification report is given below:")
print(report)


# for class 0:<br>
# &emsp;    1.precision is 90% which means that 90% of label 0 is corectly classified among total number of predicted negative           example<br>
# &emsp;    2.recall is 82% which means that 82% of label 0 is corectly classified among total number of negative example<br>
# for class 1:<br>
# &emsp;     1.precision is 84% which means that 81% of label 1 is corectly classified among total number of predicted positive           example<br>
# &emsp;    2.recall is 91% which means that 89% of label 1 is corectly classified among total number of positive example<br>

# In[216]:


print("roc-auc-score of model is {}".format(roc_auc_score(y_test,y_pred)))


# In[217]:


#ROC_AUC_CURVE IS GIVEN BELOW
curve(y_test,y_pred)


# 4.LogisticRegression

# In[218]:


train_scores=[]
test_scores=[]
for i in [0.0001, 0.001, 0.01, 0.1, 1]:
    x=LogisticRegression(C=i)
    x.fit(X_train,y_train)
    y_pred=x.predict(X_test)
    test_scores.append(accuracy_score(y_test,y_pred))
    train_scores.append(accuracy_score(y_train,x.predict(X_train)))        


# In[219]:


df=pd.DataFrame({'n':[0.0001, 0.001, 0.01, 0.1, 1],'train_score':train_scores,'test_score':test_scores})
df.columns=['n','train_score','test_score']
df

from above dataframe we can see that  train accuracy and test accuracy is closest while c=1.
# In[220]:


plt.figure(figsize=(45,14))
plt.subplot(2,2,1)
plt.plot([0.0001, 0.001, 0.01, 0.1, 1]
            ,train_scores
            ,color='blue'
            ,marker='o'
            ,markersize=5
            ,label='training accuracy')
    
plt.plot([0.0001, 0.001, 0.01, 0.1, 1]
            ,test_scores
            ,color='green'
            ,marker='x'
            ,markersize=5
            ,label='test accuracy')    
plt.xlabel('C_parameter')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.5,1])


# In[221]:


log_reg=LogisticRegression()
log_reg.fit(X_train,y_train)


# In[222]:


#predict on unseen test data 
y_pred=log_reg.predict(X_test)


# In[223]:


#accuracy of model using classsification metrics
matrix,report,score=metrics(y_test,y_pred)


# In[224]:


#confusion matrix of Random forest Classifier
print("confusion matrix is given below:")
sns.heatmap(matrix, annot=True, fmt='.2f',xticklabels = ["0","1"] , yticklabels = ["0","1"])
plt.ylabel('True class')
plt.xlabel('Predicted class')


# In[225]:


tn, fp, fn, tp = matrix.ravel()
print(" confusion matrix shows that  we correctly classified {} and misclassified {} cases out of {} ".format(tn+tp,fp+fn,tn+tp+fp+fn))


# In[226]:


print("value of false negative {}".format(fn))


# one of the most important metrics in medical cases for classification problm is false negative which represtent that observation is positive but prediction is negative .In our dataset value of False negative is quite low which is a good indicator of accuracy of our model

# In[227]:


print("We got {} accuracy on test data".format(accuracy_score(y_test,y_pred)))


# In[228]:


test_acc5=accuracy_score(y_test,y_pred)
train_acc5=accuracy_score(y_train,log_reg.predict(X_train))


# In[229]:


accuracy_score(y_train,log_reg.predict(X_train))


# In[230]:


print("classification report is given below:")
print(report)


# In[231]:


print("roc-auc-score of model is {}".format(roc_auc_score(y_test,y_pred)))


# In[232]:


#ROC_AUC_CURVE IS GIVEN BELOW
curve(y_test,y_pred)


# In[258]:


acc=pd.DataFrame({'model':['Decision Tree','Random Forest','SVM','KNN','Logistic REgression'],
                  'training_accuracy':[train_acc1,train_acc2,train_acc3,train_acc4,train_acc5],
                  'testing_accuracy':[test_acc1,test_acc2,test_acc3,test_acc4,test_acc5]})


# In[259]:


acc.set_index("model")


# In[265]:


fig,ax=plt.subplots(figsize=(10,5))
index=np.arange(5)
bar_width=0.35
training_score=ax.bar(index,acc['training_accuracy'],color='b',width=bar_width,label="train_score")
forest_score=ax.bar(index+0.37,acc['testing_accuracy'],color='r',width=bar_width,label="test_score")
ax.set_xlabel('label')
ax.set_ylabel('score')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(acc['model'])
ax.legend()


# In[261]:


plt.figure(figsize=(45,14))
plt.subplot(2,2,1)
plt.plot(acc['model']
            ,acc['training_accuracy']
            ,color='blue'
            ,marker='o'
            ,markersize=5
            ,label='training accuracy')
    
plt.plot(acc['model']
            ,acc['testing_accuracy']
            ,color='green'
            ,marker='x'
            ,markersize=5
            ,label='test accuracy')    
plt.xlabel('C_parameter')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.5,1])


# Here we can see that KNN model working really well in this dataset having accuracy of 87%.But we can improve the model accuracy by ensembling more than one model 

# #### Ensembling

# In[100]:


#import necessary library
from sklearn.ensemble import VotingClassifier
#ensemble KNN and LogisticRegression model
model_voting=VotingClassifier(estimators=[('knn',Knearest),('lr',log_reg)])
model_voting.fit(X_train,y_train)
y_pred=model_voting.predict(X_test)


# In[101]:


#accuracy of model using classsification metrics
matrix,report,score=metrics(y_test,y_pred)


# In[102]:


#confusion matrix of Random forest Classifier
print("confusion matrix is given below:")
sns.heatmap(matrix, annot=True, fmt='.2f',xticklabels = ["healthy","patient"] , yticklabels = ["healthy","patient"])
plt.ylabel('True class')
plt.xlabel('Predicted class')


# In[103]:


tn, fp, fn, tp = matrix.ravel()
print(" confusion matrix shows that  we correctly classified {} and misclassified {} cases out of {} ".format(tn+tp,fp+fn,tn+tp+fp+fn))


# In[104]:


print("value of false negative {}".format(fn))


# In[105]:


print("We got {} accuracy on test data".format(accuracy_score(y_test,y_pred)))


# In[106]:


print("classification report is given below:")
print(report)


# In[107]:


print("roc-auc-score of model is {}".format(roc_auc_score(y_test,y_pred)))


# In[108]:


#ROC_AUC_CURVE IS GIVEN BELOW
curve(y_test,y_pred)


#     Individual accuracy of KNN and Logistic Reegression was 87% . While, by applying ensemble method we can achieve upto 89 % which is small but effective improvement in accuray

# #### Bagging

# In[1116]:


from sklearn.ensemble import BaggingClassifier
#applying bagging method in SVM model
model_bagging=BaggingClassifier(base_estimator=svm,n_estimators=10,oob_score=True,random_state=76)
model_bagging.fit(X_train,y_train)


# In[1117]:


model_bagging.oob_score_


# In[1118]:


y_pred=model_bagging.predict(X_test)
accuracy_score(y_test,y_pred)


# In[1119]:


#accuracy of model using classsification metrics
matrix,report,score=metrics(y_test,y_pred)


# In[1120]:


#confusion matrix of Random forest Classifier
print("confusion matrix is given below:")
sns.heatmap(matrix, annot=True, fmt='.2f',xticklabels = ["healthy","patient"] , yticklabels = ["healthy","patient"])
plt.ylabel('True class')
plt.xlabel('Predicted class')


# In[1121]:


print("roc-auc-score of model is {}".format(roc_auc_score(y_test,y_pred)))


# In[1122]:


#ROC_AUC_CURVE IS GIVEN BELOW
curve(y_test,y_pred)


# Accuracy of SVM model was 82% while by using bagging method(create same model on different bootstrap sample of a dataset then find the best prediction among them) we reached upto 87% accuracy

# #### Boosting

# In[146]:


import xgboost as xgb
xg=xgb.XGBClassifier()


# In[152]:


gbm_param_grid = {   
    'n_estimators': list(range(1,100)),
    'max_depth': range(1, 10),
    'learning_rate': [.1,.4, .45, .5, .55, .6],
    'colsample_bytree': [.6, .7, .8, .9, 1],
    'booster':["gbtree"],
     'min_child_weight': [0.001,0.003,0.01],
}


# In[148]:


model_boosting=RandomizedSearchCV(estimator=xg,param_distributions=gbm_param_grid,scoring='accuracy',verbose=0,cv=4,n_iter=100,refit=True)
model_boosting.fit(X_train,y_train)
y_pred=model_boosting.predict(X_test)
best_param=model_boosting.best_params_


# In[157]:


xgb_classifier=xgb.XGBClassifier(max_depth=best_param['max_depth'],
                                 learning_rate=best_param['learning_rate'],
                                 min_child_weight=best_param['min_child_weight'],
                                 n_estimators=21,booster='gbtree',
                                 colsample_bytree=0.9)
xgb_classifier.fit(X_train,y_train)
y_pred=xgb_classifier.predict(X_test)
accuracy_score(y_test,y_pred)


# In[158]:


#accuracy of model using classsification metrics
matrix,report,score=metrics(y_test,y_pred)


# In[159]:


#confusion matrix of Random forest Classifier
print("confusion matrix is given below:")
sns.heatmap(matrix, annot=True, fmt='.2f',xticklabels = ["healthy","patient"] , yticklabels = ["healthy","patient"])
plt.ylabel('True class')
plt.xlabel('Predicted class')


# In[162]:


tn, fp, fn, tp = matrix.ravel()
print(" confusion matrix shows that  we correctly classified {} and misclassified {} cases out of {} ".format(tn+tp,fp+fn,tn+tp+fp+fn))


# In[163]:


print("value of false negative {}".format(fn))


# In[164]:


print("classification report is given below:")
print(report)


# In[165]:


print("roc-auc-score of model is {}".format(roc_auc_score(y_test,y_pred)))


# In[166]:


#ROC_AUC_CURVE IS GIVEN BELOW
curve(y_test,y_pred)


# CONCLUSION

# Heart Disease is one of the major concerns for society today.But this dataset is small to predict on.we have been able to get insights from our heart disease dataset and predicted  with different machine learning model.Accuracy of diiferent models are given below:<br>
#        &emsp;&emsp;1.DecisionTree Classifier 77%<br>
#        &emsp;&emsp;2.RandomForest Classifier 83%<br>
#        &emsp;&emsp;3.SupportVectorMachine   82%<br>
#        &emsp;&emsp;4.KnearestNeighbour 87%<br>
#        &emsp;&emsp;5.LogisticRegression 87%<br>
# From above list we can see that KNN is doing well with this dataset .But by using ensemble method with Knearest Neighbour and Logistic Regression model we found that the model accuracy improves upto 89%.<br>
# This can be further improved by using large dataset.<br>

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




