import pandas as pd
import numpy as np
# import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import KFold
# from sklearn.ensemble import BaggingRegressor
# from sklearn import linear_model
#tiền xử lý
data = pd.read_csv('quake.csv',index_col=0)
x = data.iloc[:,:]
# print(x)
# Y = data.iloc[:, -1]
y = data.Richter

# print("So luong phan tu trong tap:",len(x)) #so luong phan tu
# countY = len(np.unique(y))
# print("So luong phan tu nhan:",countY,",Gom:", np.unique(y))
# print("Liet ke so luong va gia tri:\n",y.value_counts()) 

#nghi thức hold-out
i = 0
for i  in range(0,10):
    X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=1.0/3,random_state=i)
    print("So luong phan tu tap test",len(X_test))
    print("So luong nhan trong tap test", len(np.unique(y_test)))
    #knn
    Mohinh_KNN = KNeighborsRegressor(n_neighbors=22)
    Mohinh_KNN.fit(X_train, y_train)
    #rung
    forest = RandomForestRegressor(n_estimators=10,random_state=42,max_depth=10,min_samples_leaf=5)
    forest.fit(X_train,y_train)
    #cay
    tree= DecisionTreeRegressor( max_depth=10, min_samples_leaf=5) 
    tree.fit(X_train, y_train)
    
    y_pred_KNN = Mohinh_KNN.predict(X_test)
    y_pred_RF = forest.predict(X_test)
    y_pred_DecTree = tree.predict(X_test)
    
    mse_KNN = mean_squared_error(y_test, y_pred_KNN)
    mse_RF = mean_squared_error(y_test, y_pred_RF)
    mse_DecTree = mean_squared_error(y_test, y_pred_DecTree)
    print('===================================')
    print('Lan lap thu ',i+1)
    print("Do chinh xac KNN:",round(mse_KNN*100,5),"%") 
    print("Do chinh xac Random Forest:",round(mse_RF*100,5),"%")
    print("Do chinh xac DecTree:",round(mse_DecTree*100,6),"%")
    print('===================================')
    print("Do chinh xac rmse KNN:",round(np.sqrt(mse_KNN)*100,5),"%")
    print("Do chinh xac rmse Random Forest:",round(np.sqrt(mse_RF)*100,5),"%")
    print("Do chinh xac rmse DecTree:",round(np.sqrt(mse_DecTree)*100,5),"%")
    
#hiển thị biểu đồ
#mse
tb = [round(mse_KNN*100,5),round(mse_RF*100,5),round(mse_DecTree*100,6)]
lb = ['knn','random_forest','dectree']
plt.bar(lb,tb)
# plt.xlabel('X label')
# plt.ylabel('Y label')
# plt.title('title')
plt.show()
#rmse
tb = [round(np.sqrt(mse_KNN)*100,5), round(np.sqrt(mse_RF)*100,5), round(np.sqrt(mse_DecTree)*100,5)]
lb = ['knn','random_forest','dectree']
plt.bar(lb,tb)
# plt.xlabel('X label')
# plt.ylabel('Y label')
# plt.title('title')
plt.show()

#nghi thức k-fold
# kfd = KFold(n_splits=10, shuffle=True, random_state=42)
# Mohinh_KNN = KNeighborsRegressor(n_neighbors=21)
# forest = RandomForestRegressor(n_estimators=10,random_state=42,max_depth=10,min_samples_leaf=10)
# entropy = DecisionTreeRegressor( max_depth=10, min_samples_leaf=5) 
# i=0
# for train_idex, test_index in kfd.split(x):
#     i = i+1
#     x_train, x_test = x.iloc[train_idex], x.iloc[test_index]
#     y_train, y_test = y.iloc[train_idex], y.iloc[test_index]
#     print('===================================')
#     print("lan lap thu",i)
#     print("X_test", len(x_test))
#     print("X_train", len(x_train))
#     print('===================================')
#     entropy.fit(x_train, y_train)
#     Mohinh_KNN.fit(x_train, y_train)
#     forest.fit(x_train, y_train)
    
#     y_pred_KNN = Mohinh_KNN.predict(x_test)
#     y_pred_RF = forest.predict(x_test)
#     y_pred_DecTree = entropy.predict(x_test)
    
#     mse_DecTree = mean_squared_error(y_test, y_pred_DecTree)
#     mse_KNN = mean_squared_error(y_test, y_pred_KNN)
#     mse_RF = mean_squared_error(y_test, y_pred_RF)
    
#     print('===================================')
#     print('Lan lap kfold thu ',i)
#     print("Do chinh xac DecTree:",mse_DecTree*100,"%") 
#     print("Do chinh xac KNN:",mse_KNN*100,"%")   
#     print("Do chinh xac Random Forest:",mse_RF*100,"%")
#     print('===================================')
#     print("Do chinh xac rmse KNN:",np.sqrt(mse_KNN)*100,"%")
#     print("Do chinh xac rmse Random Forest:",np.sqrt(mse_RF)*100,"%")
#     print("Do chinh xac rmse DecTree:",np.sqrt(mse_DecTree)*100,"%")