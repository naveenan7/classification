#!/usr/bin/env python
# coding: utf-8

# In[34]:


# importing required libraries
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import sklearn.model_selection
import sklearn.metrics
import sklearn.neural_network
from sklearn.metrics import classification_report, confusion_matrix
# import machine learning algorithms
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier


# In[4]:


# read the train and test dataset
train_data = pd.read_csv(r'C:\Users\T.Naveenan\Desktop\Reseach\Data\processed data1\TraningDataset.csv')
test_data = pd.read_csv(r'C:\Users\T.Naveenan\Desktop\Reseach\Data\processed data1\TestingDataset.csv')


# In[5]:


train_data.info()
test_data.info()


# In[6]:


# shape of the dataset
print('Shape of training data :',train_data.shape)
print('Shape of testing data :',test_data.shape)


# In[7]:


# seperate the independent and target variable on training data
train_x = train_data.drop(columns=['class'],axis=1)
train_y = train_data['class']


# In[8]:


# seperate the independent and target variable on testing data
test_x = test_data.drop(columns=['class'],axis=1)
test_y = test_data['class']


# In[ ]:


#Naive Bayes classification


# In[9]:


Naive_Bayes_model = GaussianNB()

# fit the model with the training data
Naive_Bayes_model.fit(train_x,train_y)


# In[10]:


# predict the target on the train dataset
predict_train = Naive_Bayes_model.predict(train_x)


# In[11]:


# Accuray Score on train dataset
accuracy_train = accuracy_score(train_y,predict_train)
print('accuracy_score on train dataset : ', accuracy_train)


# In[12]:


# predict the target on the test dataset
predict_test = Naive_Bayes_model.predict(test_x)
print('Target on test data',predict_test)


# In[13]:


# Accuracy Score on test dataset
accuracy_test = accuracy_score(test_y,predict_test)
print('accuracy_score on test dataset : ', accuracy_test)


# In[21]:


#Decision Tree Classifier


# In[14]:


dt_model = DecisionTreeClassifier()
dt_model.fit(train_x, train_y)


# In[15]:


y_pred = dt_model.predict(test_x)


# In[16]:


print(confusion_matrix(test_y,y_pred))
print(classification_report(test_y,y_pred))


# In[17]:


#Gradient Boosting classifier


# In[18]:


# train with Gradient Boosting algorithm
# compute the accuracy scores on train and validation sets when training with different learning rates
 
#gb_model = GradientBoostingClassifier(n_estimators=20, learning_rate = learning_rate, max_features=2, max_depth = 2, random_state = 0)
    
learning_rates = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1]
for learning_rate in learning_rates:
    gb_model = GradientBoostingClassifier(n_estimators=20, learning_rate = learning_rate, max_features=2, max_depth = 2, random_state = 0)
    gb_model.fit(train_x, train_y)
    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb_model.score(train_x, train_y)))
    print("Accuracy score (validation): {0:.3f}".format(gb_model.score(test_x, test_y)))
    print()


# In[45]:


#MLP Classifier


# In[19]:


MLP_model = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', solver='adam', 
                                                 alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, 
                                                 max_iter=1000, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, 
                                                 nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, 
                                                 n_iter_no_change=10)


# In[20]:


MLP_model.fit(train_x, train_y)


# In[21]:


y_pred = MLP_model.predict(test_x)


# In[22]:


print(confusion_matrix(test_y,y_pred))
print(classification_report(test_y,y_pred))


# In[60]:


#SVM model


# In[23]:


from sklearn.svm import SVC
svclassifier_model = SVC(kernel='linear')
svclassifier_model.fit(train_x, train_y)


# In[24]:


y_pred = svclassifier_model.predict(test_x)


# In[25]:


print(confusion_matrix(test_y,y_pred))
print(classification_report(test_y,y_pred))


# In[26]:


# polynomial kernel
svclassifier_model_poly = SVC(kernel='poly')
svclassifier_model_poly.fit(train_x, train_y)


# In[27]:


y_pred = svclassifier_model_poly.predict(test_x)


# In[28]:


print(confusion_matrix(test_y,y_pred))
print(classification_report(test_y,y_pred))


# In[29]:


#Gaussian Kernel
svclassifier_model_rbf = SVC(kernel='rbf')
svclassifier_model_rbf.fit(train_x, train_y)


# In[30]:


y_pred = svclassifier_model_rbf.predict(test_x)


# In[31]:


print(confusion_matrix(test_y,y_pred))
print(classification_report(test_y,y_pred))


# In[35]:


Classifier = [ ('mlp', MLP_model), ('svc' ,svclassifier_model_poly ), ('dt', dt_model ), ('gb' ,gb_model), ('nb' ,Naive_Bayes_model)]
final_model = VotingClassifier(estimators= Classifier, voting='hard')


# In[36]:


final_model.fit(train_x, train_y)


# In[39]:


y_pred = final_model.predict(test_x)


# In[40]:


print(confusion_matrix(test_y,y_pred))
print(classification_report(test_y,y_pred))


# In[47]:


# Saving model to disk
import pickle
pickle.dump(final_model, open('cmodel01.pkl','wb'))


# In[ ]:

# Loading model to compare the results
cmodel = pickle.load(open('cmodel01.pkl','rb'))


