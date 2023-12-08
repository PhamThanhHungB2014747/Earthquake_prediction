# import thư viện
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

### tiền xử lý
data = pd.read_csv('quake.csv')
X = data.iloc[:, data.columns != 'Richter']
Y = data.Richter
print("Thuộc tính:")
print(X)
print("Nhãn:")
print(Y)
print("So luong phan tu trong tap du lieu quake:", len(X))
countY = len(np.unique(Y))
print("So luong phan tu nhan:", countY)
print(np.unique(Y))
print(f"Liet ke so luong theo gia tri nhan:\n{Y.value_counts()}")

'''
+ Thay đổi tham số max_depth và min_samples_leaf: 
   • max_depth và min_samples_leaf lần lượt là: (3,4), (4,3), (5,2), (5,7), (7,2)
   • random_state = 0
   • Tính RMSE và chọn bộ tham số max_depth, min_samples_leaf có RMSE tốt nhất
'''
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=1.0/3,random_state=0)
#(3,4) 18.835%
tree = DecisionTreeRegressor(max_depth=3,min_samples_leaf=4,random_state=0)
#(4,3) 18.753 %
tree = DecisionTreeRegressor(max_depth=4,min_samples_leaf=3,random_state=0)
#(5,2) 18.596 %
tree = DecisionTreeRegressor(max_depth=5,min_samples_leaf=2,random_state=0)
#(5,7) 18.774 %
tree = DecisionTreeRegressor(max_depth=5,min_samples_leaf=7,random_state=0)
#(7,2) 18.875 %
tree = DecisionTreeRegressor(max_depth=7,min_samples_leaf=2,random_state=0)
tree.fit(x_train,y_train)
y_pred_tree = tree.predict(x_test)
mse_tree = mean_squared_error(y_test,y_pred_tree)
print(round(np.sqrt(mse_tree),3),"%")

'''
+ Đánh giá mô hình hold-out:
    • Lặp qua 10 lần
    • Với cây quyết định, sử dụng max_depth và min_samples_leaf có RMSE tốt nhất từ kết quả của phần trên (max_depth=5, min_samples_leaf=2), random_state=0
    • Với KNN với k = 10
    • Với rừng ngẫu nhiên, số cây bằng 10, random_state=0, max_depth=5, min_samples_leaf=2
    • So sánh RMSE của 3 thuật toán
    • Vẽ biểu đồ so sánh RMSE dùng matplotlib
'''
barwidth=0.2
i = 0
y1 = []
y2 = []
y3 = []
for i  in range(0,10):
    X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=1.0/3,random_state=i) #
    print("So luong phan tu tap test",len(X_test))
    print("So luong nhan trong tap test", len(np.unique(y_test)))
    #knn
    Mohinh_KNN = KNeighborsRegressor(n_neighbors=10)
    Mohinh_KNN.fit(X_train, y_train)
    #rung
    forest = RandomForestRegressor(n_estimators=10,random_state=0,max_depth=5,min_samples_leaf=2)
    forest.fit(X_train,y_train)
    #cay
    tree= DecisionTreeRegressor( max_depth=5, min_samples_leaf=2, random_state=0) 
    tree.fit(X_train, y_train)
    
    y_pred_KNN = Mohinh_KNN.predict(X_test)
    y_pred_RF = forest.predict(X_test)
    y_pred_DecTree = tree.predict(X_test)
    
    mse_KNN = mean_squared_error(y_test, y_pred_KNN)
    mse_RF = mean_squared_error(y_test, y_pred_RF)
    mse_DecTree = mean_squared_error(y_test, y_pred_DecTree)
    
    rmse_KNN = round(np.sqrt(mse_KNN),3)
    rmse_RF = round(np.sqrt(mse_RF),3)
    rmse_DecTree = round(np.sqrt(mse_DecTree),3)
    
    y1.append(rmse_KNN)
    y2.append(rmse_RF)
    y3.append(rmse_DecTree)
    
    print('===================================')
    print('Lan lap thu ',i+1)
    
    print("Do chinh xac rmse KNN:",rmse_KNN,"%")
    print("Do chinh xac rmse Random Forest:",rmse_RF,"%")
    print("Do chinh xac rmse DecTree:",rmse_DecTree,"%")
    
x1 = np.arange(len(y1))
x2 = [x + barwidth for x in x1]
x3 = [x + barwidth for x in x2]
plt.bar(x1, y1, width=barwidth, label='KNN')
plt.bar(x2, y2, width=barwidth, label='Random Forest')
plt.bar(x3, y3, width=barwidth, label='Decision Tree')
plt.xlabel('Lần lặp')
plt.ylabel('RMSE')
plt.title('So sánh RMSE của 3 giải thuật')
plt.xticks([r + barwidth for r in range(len(y1))],['Lần 1','Lần 2','Lần 3','Lần 4','Lần 5','Lần 6','Lần 7','Lần 8','Lần 9','Lần 10'])
plt.legend()
plt.ylim(0,0.27)
plt.show()

#test
'''
barwidth =0.2
knn = [19.147,19.495,18.796,19.198,19.33,20.109,19.7,20.272,19.988,19.249]
rf = [18.348,18.686,18.546,18.993,18.425,19.09,18.7,19.96,19.087,18.889]
tree = [18.596,18.879,18.956,19.47,18.962,19.008,18.894,20.114,19.549,19.225]
br1 = np.arange(len(knn))
br2 = [x + barwidth for x in br1]
br3 = [x + barwidth for x in br2]
plt.bar(br1,knn,width=barwidth,edgecolor='grey',label='knn')
plt.bar(br2,rf,width=barwidth,edgecolor='grey',label='rf')
plt.bar(br3,tree,width=barwidth,edgecolor='grey',label='tree')
plt.xlabel('Lần lặp')
plt.ylabel('RMSE')
plt.title('So sánh RMSE của 3 giải thuật')
plt.xticks([r + barwidth for r in range(len(knn))],['lần 1','lần 2','lần 3','lần 4','lần 5','lần 6','lần 7','lần 8','lần 9','lần 10'])
plt.legend()
plt.show()
'''

'''
# nghi thức k-fold
kfd = KFold(n_splits=10, shuffle=True, random_state=0)
Mohinh_KNN = KNeighborsRegressor(n_neighbors=10)
forest = RandomForestRegressor(n_estimators=10,random_state=0,max_depth=5,min_samples_leaf=2)
tree = DecisionTreeRegressor( max_depth=5, min_samples_leaf=2) 
i=0
y1 = []
y2 = []
y3 = []
for train_idex, test_index in kfd.split(X):
    i = i+1
    x_train, x_test = X.iloc[train_idex], X.iloc[test_index]
    y_train, y_test = Y.iloc[train_idex], Y.iloc[test_index]
    print('===================================')
    print("lan lap thu",i)
    print("X_test", len(x_test))
    print("X_train", len(x_train))
    print('===================================')
    tree.fit(x_train, y_train)
    Mohinh_KNN.fit(x_train, y_train)
    forest.fit(x_train, y_train)
    
    y_pred_KNN = Mohinh_KNN.predict(x_test)
    y_pred_RF = forest.predict(x_test)
    y_pred_DecTree = tree.predict(x_test)
    
    mse_DecTree = mean_squared_error(y_test, y_pred_DecTree)
    mse_KNN = mean_squared_error(y_test, y_pred_KNN)
    mse_RF = mean_squared_error(y_test, y_pred_RF)
    
    rmse_KNN = round(np.sqrt(mse_KNN),3)
    rmse_RF = round(np.sqrt(mse_RF),3)
    rmse_DecTree = round(np.sqrt(mse_DecTree),3)
    
    y1.append(rmse_KNN)
    y2.append(rmse_RF)
    y3.append(rmse_DecTree)
    
    print('===================================')
    print('Lan lap kfold thu ',i)
    
    print("Do chinh xac rmse KNN:",rmse_KNN,"%")
    print("Do chinh xac rmse Random Forest:",rmse_RF,"%")
    print("Do chinh xac rmse DecTree:",rmse_DecTree,"%")

x1 = np.arange(len(y1))
x2 = [x + 0.2 for x in x1]
x3 = [x + 0.2 for x in x2]
plt.bar(x1, y1, width=0.2, label='KNN')
plt.bar(x2, y2, width=0.2, label='Random Forest')
plt.bar(x3, y3, width=0.2, label='Decision Tree')
plt.xlabel('Lần lặp')
plt.ylabel('RMSE')
plt.title('So sánh RMSE của 3 giải thuật')
plt.xticks([r + 0.2 for r in range(len(y1))],['Lần 1','Lần 2','Lần 3','Lần 4','Lần 5','Lần 6','Lần 7','Lần 8','Lần 9','Lần 10'])
plt.legend()
plt.ylim(0,0.27)
plt.show()
'''
