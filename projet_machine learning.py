# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 14:13:20 2025

@author: HP
"""

import pandas as pd 
import time 
from sklearn.preprocessing import LabelEncoder 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
# Fonction de mesure de la durée 
def measure_time(func): 
start_time = time.time() 
result = func 
end_time = time.time() 
execution_time = end_time - start_time 
print(f"Execution time: {execution_time} seconds") 
return result 
import pandas as pd 
# Charger les données 
data 
pd.read_csv("C:/Users/HP/Desktop/MIASHS/TD_Guillaume/Evaluation/bankmarketing.csv") 
# Afficher les premières lignes 
print(data.head()) 
#%% 
# Informations générales 
print(data.info()) 
# Distribution de la variable cible 
print(data["y"].value_counts()) 
#%% 
= 
from sklearn.preprocessing import LabelEncoder 
# Encodage de la variable cible 
label_encoder = LabelEncoder() 
data["y"] = label_encoder.fit_transform(data["y"]) 
# Séparation des features et de la cible 
X = data.drop("y", axis=1) 
y = data["y"] 
# Encodage des variables catégorielles (si nécessaire) 
X = pd.get_dummies(X, drop_first=True) 
#%% 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
# Séparation des données en ensembles d'entraînement et de test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 
# k-NN avec k = 5 
knn = KNeighborsClassifier(n_neighbors=5) 
knn.fit(X_train, y_train) 
# Prédiction et évaluation 
y_pred = knn.predict(X_test) 
accuracy = accuracy_score(y_test, y_pred) 
print(f"Accuracy k-NN: {accuracy}") 
measure_time(accuracy)  
from sklearn.neighbors import NearestNeighbors 
# Réduction de l'ensemble d'entraînement avec CNN 
cnn = NearestNeighbors(n_neighbors=1) 
cnn.fit(X_train) 
_, indices = cnn.kneighbors(X_train) 
# Sélection des points conservés 
X_train_cnn = X_train.iloc[indices.flatten()] 
y_train_cnn = y_train.iloc[indices.flatten()] 
# k-NN sur l'ensemble condensé 
knn_cnn = KNeighborsClassifier(n_neighbors=5) 
knn_cnn.fit(X_train_cnn, y_train_cnn) 
# Prédiction et évaluation 
y_pred_cnn = knn_cnn.predict(X_test) 
accuracy_cnn = accuracy_score(y_test, y_pred_cnn) 
print(f"Accuracy CNN: {accuracy_cnn}") 
measure_time(accuracy_cnn)  
#%% 
from sklearn.svm import SVC 
# SVM linéaire 
svm = SVC(kernel="linear") 
svm.fit(X_train, y_train) 
# Prédiction et évaluation 
y_pred_svm = svm.predict(X_test) 
accuracy_svm = accuracy_score(y_test, y_pred_svm) 
print(f"Accuracy SVM linéaire: {accuracy_svm}") 
measure_time(accuracy_svm)  
from sklearn.linear_model import LogisticRegression 
# Régression logistique 
logit = LogisticRegression(max_iter=1000) 
logit.fit(X_train, y_train) 
# Prédiction et évaluation 
y_pred_logit = logit.predict(X_test) 
accuracy_logit = accuracy_score(y_test, y_pred_logit) 
print(f"Accuracy Régression logistique: {accuracy_logit}") 
measure_time(accuracy_logit)  
# SVM avec noyau gaussien 
svm_gaussian = SVC(kernel="rbf") 
svm_gaussian.fit(X_train, y_train) 
# Prédiction et évaluation 
y_pred_gaussian = svm_gaussian.predict(X_test) 
accuracy_gaussian = accuracy_score(y_test, y_pred_gaussian) 
print(f"Accuracy SVM gaussien: {accuracy_gaussian}") 
measure_time(accuracy_gaussian)  
#%% 
from sklearn.tree import DecisionTreeClassifier 
# Arbre de décision 
tree = DecisionTreeClassifier() 
tree.fit(X_train, y_train) 
# Prédiction et évaluation 
y_pred_tree = tree.predict(X_test) 
accuracy_tree = accuracy_score(y_test, y_pred_tree) 
print(f"Accuracy Arbre de décision: {accuracy_tree}") 
measure_time(accuracy_tree) 
from sklearn.ensemble import RandomForestClassifier 
# Forêt aléatoire 
rf = RandomForestClassifier() 
rf.fit(X_train, y_train) 
# Prédiction et évaluation 
y_pred_rf = rf.predict(X_test) 
accuracy_rf = accuracy_score(y_test, y_pred_rf) 
print(f"Accuracy Forêt aléatoire: {accuracy_rf}") 
measure_time(accuracy_rf)  
from sklearn.ensemble import AdaBoostClassifier 
# Adaboost 
adaboost = AdaBoostClassifier() 
adaboost.fit(X_train, y_train) 
# Prédiction et évaluation 
y_pred_adaboost = adaboost.predict(X_test) 
accuracy_adaboost = accuracy_score(y_test, y_pred_adaboost) 
print(f"Accuracy Adaboost: {accuracy_adaboost}") 
measure_time(accuracy_adaboost)