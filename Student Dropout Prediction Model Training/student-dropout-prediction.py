#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[19]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression,Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay


# # Load Data

# In[20]:


df = pd.read_csv("data.csv",sep=";")
df


# # Data Cleaning, Information & Visualization

# In[21]:


df.info()


# In[22]:


df.isnull().sum()


# In[23]:


df.shape


# In[24]:


df.size


# In[25]:


df.describe().T


# In[26]:


df['Target'].value_counts()


# In[27]:


df['Target'] = LabelEncoder().fit_transform(df['Target'])


# In[28]:


df['Target'].value_counts()


# In[29]:


plt.figure(figsize=(5, 10))
sns.distplot(df['Target'], color = "Blue")


# In[30]:


plt.figure(figsize=(5, 10))
sns.countplot(data = df, x="Target").set_title('Target')


# In[31]:


plt.figure(figsize=(8, 8))
plt.title("Education Status")
plt.pie(df['Target'].value_counts(), labels = ['Graduate', 'Dropout', 'Enrolled'], explode = (0.1, 0.1, 0.0), autopct='%1.2f%%', shadow = True)
plt.legend( loc = 'lower right')


# In[32]:


plt.figure(figsize=(8, 8))
plt.title("Gender")
plt.pie(df['Gender'].value_counts(), labels = ['Male', 'Female'], explode = (0.1, 0.0), autopct='%1.2f%%', shadow = True)
plt.legend( loc = 'lower right')


# In[33]:


plt.figure(figsize=(20, 45))

for i in range(0, 35):
    plt.subplot(12,3,i+1)
    sns.distplot(df.iloc[:, i], color='blue')
    plt.grid()


# In[36]:


#feature selection
corr_matrix = df.corr(method="pearson")
plt.figure(figsize=(10, 10)) 
sns.heatmap(corr_matrix, vmin=-1., vmax=1., annot=False, fmt='.2f', cmap="YlGnBu", cbar=True, linewidths=0.5)
plt.title("Pearson correlation")
plt.show()


# In[37]:


["Tuition fees up to date","Curricular units 1st sem (approved)","Curricular units 1st sem (grade)","Curricular units 2nd sem (approved)","Curricular units 2nd sem (grade)"]
corr_matrix["Target"]


# ## **Assertion**
# 
# ### As we are predicting whether a student will dropout or not so, the number of "Enrolled" student is irrelevant. We only need to know whether a student graduated or dropedout. So, we are dropping the "Enrolled" values and going forward with "Graduate" & "Dropout" values.

# In[38]:


df.drop(df[df['Target'] == 1].index, inplace = True)
df


# In[39]:


df['Dropout'] = df['Target'].apply(lambda x: 1 if x==0 else 0)
df


# In[40]:


plt.figure(figsize=(5, 10))
sns.distplot(df['Dropout'], color = "red")


# In[41]:


plt.figure(figsize=(8, 8))
plt.title("Dropout Status")
plt.pie(df['Dropout'].value_counts(),  labels = ['Non-Dropout', 'Dropout'], explode = (0.2, 0.0), autopct='%1.2f%%', shadow = True)
plt.legend( loc = 'lower right')


# # Standard Scaling the Data

# In[42]:


x = df.iloc[:, :36].values
#x = df[["Tuition fees up to date","Curricular units 1st sem (approved)","Curricular units 1st sem (grade)","Curricular units 2nd sem (approved)","Curricular units 2nd sem (grade)"]].values
print(x)
x = StandardScaler().fit_transform(x)
x


# In[43]:


y = df['Dropout'].values
y


# # Train & Test Splitting the Data

# In[44]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)


# # Function to Measure Performance

# In[45]:


def perform(y_pred):
    print("Precision : ", precision_score(y_test, y_pred, average = 'micro'))
    print("Recall : ", recall_score(y_test, y_pred, average = 'micro'))
    print("Accuracy : ", accuracy_score(y_test, y_pred))
    print("F1 Score : ", f1_score(y_test, y_pred, average = 'micro'))
    cm = confusion_matrix(y_test, y_pred)
    print("\n", cm)
    print("\n")
    print("**"*27 + "\n" + " "* 16 + "Classification Report\n" + "**"*27)
    print(classification_report(y_test, y_pred))
    print("**"*27+"\n")
    
    cm = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=['Non-Dropout', 'Dropout'])
    cm.plot()


# # Gaussian Naive Bayes

# In[46]:


model_nb = GaussianNB()
model_nb.fit(x_train, y_train)


# In[47]:


y_pred_nb = model_nb.predict(x_test)


# In[48]:


perform(y_pred_nb)


# # Logistic Regression

# In[49]:


model_lr = LogisticRegression()
model_lr.fit(x_train, y_train)


# In[50]:


y_pred_lr = model_lr.predict(x_test)


# In[51]:


perform(y_pred_lr)


# # Random Forest

# In[65]:


model_rf = RandomForestClassifier(n_estimators=500,criterion='entropy')
model_rf.fit(x_train, y_train)


# In[66]:


y_pred_rf = model_rf.predict(x_test)


# In[67]:


perform(y_pred_rf)


# # Support Vector Classifier

# In[114]:


model_svc = SVC(C=0.1,kernel='linear')
model_svc.fit(x_train, y_train)


# In[115]:


y_pred_svc = model_svc.predict(x_test)


# In[116]:


perform(y_pred_svc)


# # Perceptron

# In[150]:


model_mlp = Perceptron(alpha=0.001,l1_ratio=0.5,max_iter=100)
model_mlp.fit(x_train, y_train)


# In[151]:


y_pred_mlp = model_mlp.predict(x_test)


# In[152]:


perform(y_pred_mlp)


# # KNN Classifier

# In[100]:


error = []

# Calculating MAE error for K values between 1 and 39
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    pred_i = knn.predict(x_test)
    accuracy = accuracy_score(y_test, pred_i)
    error.append(accuracy)


# In[101]:


plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', 
         linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
         
plt.title('K Value accuracy')
plt.xlabel('K Value')
plt.ylabel('Accuracy')


# In[153]:


model_knn = KNeighborsClassifier(n_neighbors=3)
model_knn.fit(x_train, y_train)


# In[154]:


y_pred_knn = model_knn.predict(x_test)


# In[156]:


perform(y_pred_knn)


# # Comparison

# In[160]:


pred=[y_pred_nb,y_pred_lr,y_pred_rf,y_pred_svc,y_pred_mlp,y_pred_knn]
acc=[]
classifiers=["NaiveBayes","Logistic Regression","RandomForest","Support Vector Classier","Perceptron","KNN"]
for i in pred:
    temp=accuracy_score(y_test, i)
    acc.append(temp)

plt.barh(classifiers, acc)

# Add labels and title
plt.ylabel('classifiers')
plt.xlabel('Accuracy')
plt.title('Comparison')
plt.show()

    


# In[ ]:




