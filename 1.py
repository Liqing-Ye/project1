#!/usr/bin/env python
# coding: utf-8

# In[20]:


### project_1 

import time #import time fuction to record the running time when it trains each classifier model
import pandas as pd # for data processing
from sklearn.model_selection import train_test_split # import train_test_split function
from sklearn.metrics import accuracy_score # for accuracy calculation

# Train Dataset preprocessing
col_names = ["gameId","creationTime","gameDuration","seasonId","winner","firstBlood","firstTower",
             "firstInhibitor","firstBaron","firstDragon","firstRiftHerald",
             "t1_towerKills","t1_inhibitorKills","t1_baronKills","t1_dragonKills",
             "t1_riftHeraldKills","t2_towerKills","t2_inhibitorKills","t2_baronKills",
             "t2_dragonKills","t2_riftHeraldKills"]
feature_cols=["firstBlood","firstTower",
             "firstInhibitor","firstBaron","firstDragon","firstRiftHerald",
             "t1_towerKills","t1_inhibitorKills","t1_baronKills","t1_dragonKills",
             "t1_riftHeraldKills","t2_towerKills","t2_inhibitorKills","t2_baronKills",
             "t2_dragonKills","t2_riftHeraldKills"]
dataset=pd.read_csv("G:\course_2_1\BigData\project_1\\test_set.csv",header=None, names=col_names) # load the training dataset
dataset=dataset.iloc[1:]
X=dataset[feature_cols].values # extract the feather set, convert it to values
Y=dataset.winner.values  # extract the label set, convert it to values
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=1) # split the dataset into training set and testing one

# Test Dataset preprocessing
testnew_dataset=pd.read_csv("G:\course_2_1\BigData\project_1\\new_data.csv",header=None, names=col_names) # load the testing dataset
testnew_dataset=testnew_dataset.iloc[1:]
X_testnew=testnew_dataset[feature_cols].values
Y_testnew=testnew_dataset.winner.values


### < summary >
### DT classifier, training the train data set
### < /summary >
### < return > < /return >
def _DT():
    from sklearn.tree import DecisionTreeClassifier #import Decision Tree Classifier
    startDTtime=time.time() # training start time for the DT classifier
    clf=DecisionTreeClassifier(criterion="entropy", max_depth=4)
    clf=clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    print(clf)
    print("Training accuracy of the DT classifier :",accuracy_score(y_test,y_pred))
    endDTtime=time.time() # training end timme for the DT classifier
    print("Training time of the DT classifier: %s Seconds"%(endDTtime-startDTtime))
    return clf

### < summary >
### ANN classifier, training the train data set
### < /summary >
### < param name = X_train,X_test,y_train,y_test > < /param >
### < return > < /return >
def _ANN(X_train,X_test,y_train,y_test):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    X_train=X_train.astype(float) # numpy, convert data to float type
    X_train = torch.FloatTensor(X_train)
    X_test=X_test.astype(float)
    X_test = torch.FloatTensor(X_test)
    y_train=y_train.astype(float)
    y_train = torch.LongTensor(y_train)
    y_test=y_test.astype(float)
    y_test = torch.LongTensor(y_test)

    startANNtime=time.time() # training start time for the ANN classifier
    class ANN(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(in_features=16, out_features=264)
            self.output = nn.Linear(in_features=264, out_features=3)
        def forward(self, x):
            x = torch.sigmoid(self.fc1(x))
            x = self.output(x)
            x = F.softmax(x,dim=1)
            return x

    model = ANN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    epochs = 100
    loss_arr = []
    for i in range(epochs):
        y_hat = model.forward(X_train)
        loss = criterion(y_hat, y_train)
        loss_arr.append(loss)
        #if i % 2 == 0:
            #print(f'Epoch: {i} Loss: {loss}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    predict_out = model(X_test)
    _,predict_y = torch.max(predict_out, 1)

    #print("\nThe predict of training set by ANN classifier: ",predict_y)
    print("\n",model)
    print("Training accuracy of the ANN classifier :", accuracy_score(y_test, predict_y) )
    endANNtime=time.time() # training end timme for the ANN classifier
    print("Training time of the ANN classifier: %s Seconds"%(endANNtime-startANNtime))
    return model

### < summary >
### MLP
### < /summary >
### < return > < /return >
def _MLP():
    startMLPtime=time.time() # training start time for the MLP classifier
    MLPparam_grid={
        'activation':['relu'],#['identity','logistic','tanh','relu'], 
        'hidden_layer_sizes':[(100,)],
        #'solver':['sgd','adam'],#'lbfgs',
        #'alpha':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
        'learning_rate':['invscaling'], #'constant','invscaling','adaptive'
        'max_iter':[100]
    }

    from sklearn.neural_network import MLPClassifier
    cls_MLP=MLPClassifier()
    from sklearn.model_selection import GridSearchCV
    MLP_gridSearch=GridSearchCV(cls_MLP,MLPparam_grid)
    MLP_gridSearch.fit(X_train,y_train)
    MLP_best_params=MLP_gridSearch.best_estimator_
    MLPpred=MLP_gridSearch.predict(X_test) 
    MLPprobability=MLP_gridSearch.predict_proba(X_test)  
    MLPaccuracy=accuracy_score(y_test,MLPpred)
    print('\n',MLP_gridSearch)
    print('\nTraining accuracy of the MLP classifier :',MLPaccuracy) 
    endMLPtime=time.time() # training end timme for the MLP classifier
    print("Training time of the MLP classifier: %s Seconds"%(endMLPtime-startMLPtime))
    return MLP_gridSearch

def _KNN():
    from sklearn.model_selection import GridSearchCV
    startKNNtime=time.time() # training start time for the KNN classifier
    KNNparam_grid={
        'weights':['uniform'], #'distance'
        'n_neighbors':[5],#[i for i in range(1,11)],
        'p':[2]#[j for j in range(1,6)]
    }

    from sklearn.neighbors import KNeighborsClassifier
    cls_KNN=KNeighborsClassifier()
    KNN_gridSearch=GridSearchCV(cls_KNN,KNNparam_grid)
    KNN_gridSearch.fit(X_train,y_train)
    KNN_best_params=KNN_gridSearch.best_estimator_
    KNNpred=KNN_gridSearch.predict(X_test) 
    #KNNprobability=cls_KNN.predict_proba(_data,probability=True)
    from sklearn.metrics import accuracy_score
    KNNaccuracy=accuracy_score(y_test,KNNpred)
    print('\n',cls_KNN)
    print('KNNaccuracy: ',KNNaccuracy)
    
    endKNNtime=time.time() # training end timme for the KNN classifier
    print("Training time of the KNN classifier: %s Seconds"%(endKNNtime-startKNNtime))
    return KNN_gridSearch


### < summary >
### testing the test data set by DT classifier
### evalute the accuracy
### < /summary >
### < return > < /return >
def pred_byDT(clf):
    testnew_pred=clf.predict(X_testnew)
    print("\nTesting accuracy of the DT classifier :",accuracy_score(Y_testnew,testnew_pred))
    return

### < summary >
### testing the test data set by ANN classifier
### evalute the accuracy
### < /summary >
### < param name = X_testnew,Y_testnew > < /param >
### < return > < /return >
def pred_byANN(X_testnew,Y_testnew,model):
    import torch
    X_testnew=X_testnew.astype(float) 
    X_testnew = torch.FloatTensor(X_testnew)
    Y_testnew=Y_testnew.astype(float)
    Y_testnew = torch.FloatTensor(Y_testnew)

    test_predict_out = model(X_testnew)
    test_,test_predict_ANN = torch.max(test_predict_out, 1)

    #print("\nThe predict of test set by ANN classifier: ",test_predict_ANN)
    print("Testing accuracy of the ANN classifier : ", accuracy_score(Y_testnew, test_predict_ANN) )
    return

def pred_byMLP(MLP_gridSearch):
    testnew_pred=MLP_gridSearch.predict(X_testnew) # MLPpred=MLP_gridSearch.predict(X_test) 
    print("Testing accuracy of the MLP classifier :",accuracy_score(Y_testnew,testnew_pred))
    return

def pred_byKNN(KNN_gridSearch):
    testnew_pred=KNN_gridSearch.predict(X_testnew)
    print("Testing accuracy of the KNN classifier :",accuracy_score(Y_testnew,testnew_pred))
    return

clf=_DT()
pred_byDT(clf)
model=_ANN(X_train,X_test,y_train,y_test)
pred_byANN(X_testnew,Y_testnew,model)
MLP_gridSearch=_MLP()
pred_byMLP(MLP_gridSearch)
KNN_gridSearch=_KNN()
pred_byKNN(KNN_gridSearch)


# In[ ]:




