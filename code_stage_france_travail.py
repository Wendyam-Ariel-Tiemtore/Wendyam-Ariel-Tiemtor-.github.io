# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 13:48:35 2025

@author: HP
"""

#%% Importation des packages pour l'analyse 
import pandas as pd 
from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt 
import numpy as np 
from scipy.stats import chi2_contingency, fisher_exact 
import seaborn as sns 
#%% Pour permettre d'afficher complètement les résultats dans la console 
pd.set_option('display.max_columns', None)  # Affiche toutes les colonnes 
pd.set_option('display.max_rows', None)    # Affiche toutes les lignes 
#%% Chargement des données 
df = pd.read_excel("C:/Users/IWTI4850/France Travail/ARA - DR - Statistiques, études et 
évaluations - General/STAGE 2025/TRAVAIL WENDYAM/Analyse ETS/base_py.xlsx") 
#%% test statistiques 
def test_nominal_association(var1, var2): 
# Création du tableau de contingence 
contingency_table = pd.crosstab(var1, var2) 
# Test du Chi-carré 
chi2, p, dof, expected = chi2_contingency(contingency_table) 
# V de Cramer 
n = contingency_table.sum().sum() 
phi = np.sqrt(chi2 / n) 
r, k = contingency_table.shape 
v = np.sqrt(phi**2 / min((k-1), (r-1))) 
50 
# Test exact de Fisher (si tableau 2x2) 
fisher_p = np.nan 
if contingency_table.shape == (2, 2): 
_, fisher_p = fisher_exact(contingency_table) 
return { 
'Chi2': chi2, 
'p-value': p, 
'V de Cramer': v, 
'Degrés de liberté': dof 
} 
#%% Recodage des variables en variable numérique 
# Suppression des doublons dans le recodage 
df['dpae_recodé'] = df['DPAE_ets'].apply(lambda x: 1 if x == 'OUI' else 0) 
df['offre_recodé'] = df['Offre_ets'].apply(lambda x: 1 if x == 'OUI' else 0) 
df['ent_recodé'] = df['ent_ets'].apply(lambda x: 1 if x == 'mono établissement' else 2) 
# Mise à jour du mapping des tranches d'effectif 
tranches_effectif_mapping = { 
'0 salarié': 0, 
'1 à 9 salariés': 1, 
'10 à 19 salariés': 2, 
'20 à 49 salariés': 3, 
'50 à 99 salariés': 4, 
'100 salariés et plus': 5 
} 
df['tranche_recodé'] = df['Tranche effectif créée'].map(tranches_effectif_mapping) 
# Conversion de la densité en numérique (en supposant que c'est une chaîne avec des chiffres) 
51 
df['Densité'] = pd.to_numeric(df['Densité'], errors='coerce') 
#%% Teste les liaisons entre toutes les paires de variables qualitatives nominales 
qual_vars = ['Densité', 'tranche_recodé', 'dpae_recodé', 'offre_recodé', 'ent_recodé','BE', 
'NAF_A88'] 
results = {} 
for i in range(len(qual_vars)): 
for j in range(i + 1, len(qual_vars)): 
var1 = qual_vars[i] 
var2 = qual_vars[j] 
results[f"{var1} vs {var2}"] = test_nominal_association(df[var1], df[var2]) 
# Affichage des résultats 
print("Résultats des tests de liaison entre variables qualitatives nominales:") 
results_df = pd.DataFrame(results).T 
print(results_df) 
#%% Fonction pour calculer le V de Cramer entre deux variables qualitatives 
def cramers_v(var1, var2): 
contingency_table = pd.crosstab(var1, var2) 
n = contingency_table.sum().sum() 
chi2, _, _, _ = chi2_contingency(contingency_table) 
phi = np.sqrt(chi2 / n) 
r, k = contingency_table.shape 
return np.sqrt(phi**2 / min((k-1), (r-1))) 
#%% Création de la matrice de corrélation 
correlation_matrix = pd.DataFrame(index=qual_vars, columns=qual_vars) 
52 
for i in range(len(qual_vars)): 
for j in range(len(qual_vars)): 
if i == j: 
correlation_matrix.loc[qual_vars[i], qual_vars[j]] = 1.0  # Diagonale = 1 
elif j > i: 
# Utiliser le résultat déjà calculé pour éviter les doublons 
correlation_matrix.loc[qual_vars[i], qual_vars[j]] = results_df.loc[f"{qual_vars[i]} vs 
{qual_vars[j]}", 'V de Cramer'] 
else: 
# Symétrie de la matrice 
correlation_matrix.loc[qual_vars[i], qual_vars[j]] = 
correlation_matrix.loc[qual_vars[j], qual_vars[i]] 
# Affichage de la matrice de corrélation 
print("\nMatrice de corrélation (V de Cramer) entre variables qualitatives nominales:") 
print(correlation_matrix) 
# Visualisation de la matrice de corrélation 
plt.figure(figsize=(10, 8)) 
sns.heatmap(correlation_matrix.astype(float), annot=True, cmap='coolwarm', vmin=0, 
vmax=1) 
plt.title("Matrice de corrélation (V de Cramer) entre variables qualitatives nominales") 
plt.show() 
#%% Détermination du nombre de facteurs 
# Sélection des variables quantitatives et encodage des variables catégorielles 
quant_vars = ['Densité', 'tranche_recodé', 'dpae_recodé', 'offre_recodé', 'ent_recodé','BE', 
'NAF_A88'] 
53 
df_quant = df[quant_vars] 
# Suppression des lignes avec des valeurs manquantes 
df_quant = df_quant.dropna() 
inerties = [] 
for k in range(1, 11): 
kmeans = KMeans(n_clusters=k, random_state=0) 
kmeans.fit(df_quant) 
inerties.append(kmeans.inertia_) 
# Tracer le graphique de l'inertie 
plt.figure(figsize=(8, 6)) 
plt.plot(range(1, 11), inerties, marker='o') 
plt.xlabel('Nombre de clusters') 
plt.ylabel('Inertie') 
plt.title('Méthode du coude pour déterminer le nombre optimal de clusters') 
plt.grid(True) 
plt.show() 
#%% Analyse factorielle (ACP) 
# Standardisation 
scaler = StandardScaler() 
df_std = scaler.fit_transform(df_quant) 
#%% ACP avec 6 composantes principales 
pca = PCA(n_components=6) 
pca.fit(df_std) 
54 
#Scores des composantes principales 
factor_scores = pca.transform(df_std) 
df.loc[df_quant.index, 'FAC1_1'] = factor_scores[:,0] 
df.loc[df_quant.index, 'FAC2_1'] = factor_scores[:,1] 
df.loc[df_quant.index, 'FAC3_1'] = factor_scores[:,2] 
df.loc[df_quant.index, 'FAC4_1'] = factor_scores[:,3] 
df.loc[df_quant.index, 'FAC5_1'] = factor_scores[:,4] 
df.loc[df_quant.index, 'FAC6_1'] = factor_scores[:,5] 
# Affichage des résultats 
print("Variances expliquées:") 
print(pca.explained_variance_ratio_) 
#%% correlation entre les composantes principales et les variables standardisées 
correlations = pd.DataFrame(df_std, 
columns=df_quant.columns).corrwith(pd.DataFrame(factor_scores, columns=['FAC1_1', 
'FAC2_1', 'FAC3_1', 'FAC4_1', 'FAC5_1', 'FAC6_1'])) 
# Création d'une matrice de corrélation pour une meilleure visualisation 
correlation_matrix = pd.DataFrame(index=df_quant.columns, columns=['FAC1_1', 'FAC2_1', 
'FAC3_1', 'FAC4_1', 'FAC5_1', 'FAC6_1']) 
for i, component in enumerate(['FAC1_1', 'FAC2_1', 'FAC3_1', 'FAC4_1', 'FAC5_1', 
'FAC6_1']): 
for var in df_quant.columns: 
correlation_matrix.loc[var, component] = np.corrcoef(df_std[:, i], factor_scores[:, i])[0, 
1] 
# Visualisation de la matrice de corrélation 
plt.figure(figsize=(10, 8)) 
55 
sns.heatmap(correlation_matrix.astype(float), annot=True, cmap='coolwarm', vmin=-1, 
vmax=1) 
plt.title("Matrice de corrélation entre les composantes principales et les variables 
quantitatives") 
plt.show() 
#%% Clustering K-means sur les scores factoriels 
kmeans = KMeans(n_clusters=6, max_iter=100, tol=1e-5, random_state=42) 
df.loc[df_quant.index, 'CLUSTER_ID'] = kmeans.fit_predict(factor_scores) 
#%% Tableaux de profils 
print("\nCroisement Cluster x offre_recodé:") 
print(pd.crosstab(df.loc[df_quant.index, 'CLUSTER_ID'], df.loc[df_quant.index, 
'offre_recodé'], 
margins=True, normalize='index')) 
print("\nCroisement Cluster x dpae_recodé:") 
print(pd.crosstab(df.loc[df_quant.index, 'CLUSTER_ID'], df.loc[df_quant.index, 
'dpae_recodé'], 
margins=True, normalize='index')) 
#%% Statistiques descriptives par cluster 
print("\nMoyennes par cluster:") 
print(df.loc[df_quant.index].groupby('CLUSTER_ID')['Densité', 'tranche_recodé', 
'dpae_recodé', 'offre_recodé', 'ent_recodé','BE', 'NAF_A88']) 
#%% 6. Profilage des clusters 
print("\nCroisement Cluster x Baasin d'emploi':") 
print(pd.crosstab(df.loc[df_quant.index, 'CLUSTER_ID'], df.loc[df_quant.index, 'BE'], 
normalize='index')) 
print("\nCroisement Cluster x Code NAF A88:") 
56 
print(pd.crosstab(df.loc[df_quant.index, 'CLUSTER_ID'], df.loc[df_quant.index, 'NAF_A88'], 
normalize='index')) 
print("\nCroisement Cluster x densité:") 
print(pd.crosstab(df.loc[df_quant.index, 'CLUSTER_ID'], df.loc[df_quant.index, 'Densité'], 
normalize='index')) 
print("\nCroisement Cluster x tranche effectif salarié:") 
print(pd.crosstab(df.loc[df_quant.index, 'CLUSTER_ID'], df.loc[df_quant.index, 
'tranche_recodé'], normalize='index')) 
print("\nCroisement Cluster x nombre d'établissement par entreprise':") 
print(pd.crosstab(df.loc[df_quant.index, 'CLUSTER_ID'], df.loc[df_quant.index, 
'ent_recodé'], normalize='index')) 
#%% Visualisation des établissements 
scatter = plt.scatter(df.loc[df_quant.index, 'FAC1_1'], df.loc[df_quant.index, 'FAC2_1'], 
c=df.loc[df_quant.index, 'CLUSTER_ID'], cmap='viridis') 
plt.xlabel('Facteur 1') 
plt.ylabel('Facteur 2') 
plt.title('Clusters dans l\'espace factoriel') 
# Ajout de la légende 
legend = scatter.legend_elements() 
plt.legend(legend[0], ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3','Cluster 4','Cluster 5'], 
title="Clusters") 
plt.show() 
#%% test de Chi2 
from scipy.stats import chi2_contingency 
57 
contingence = pd.crosstab(df.loc[df_quant.index, 'CLUSTER_ID'], 
df.loc[df_quant.index, 'Densité']) 
chi2, p, dof, expected = chi2_contingency(contingence) 
print("Chi2 =", chi2, "p-value =", p) 
#%% 
contingence = pd.crosstab(df.loc[df_quant.index, 'CLUSTER_ID'], 
df.loc[df_quant.index, 'tranche_recodé']) 
chi2, p, dof, expected = chi2_contingency(contingence) 
print("Chi2 =", chi2, "p-value =", p) 
#%% 
contingence = pd.crosstab(df.loc[df_quant.index, 'CLUSTER_ID'], 
df.loc[df_quant.index, 'offre_recodé']) 
chi2, p, dof, expected = chi2_contingency(contingence) 
print("Chi2 =", chi2, "p-value =", p) 
#%% 
contingence = pd.crosstab(df.loc[df_quant.index, 'CLUSTER_ID'], 
df.loc[df_quant.index, 'dpae_recodé']) 
chi2, p, dof, expected = chi2_contingency(contingence) 
print("Chi2 =", chi2, "p-value =", p) 
#%% 
contingence = pd.crosstab(df.loc[df_quant.index, 'CLUSTER_ID'], 
df.loc[df_quant.index, 'ent_recodé']) 
chi2, p, dof, expected = chi2_contingency(contingence) 
print("Chi2 =", chi2, "p-value =", p) 
#%% 
contingence = pd.crosstab(df.loc[df_quant.index, 'CLUSTER_ID'], 
df.loc[df_quant.index, 'BE']) 
chi2, p, dof, expected = chi2_contingency(contingence) 
print("Chi2 =", chi2, "p-value =", p) 
58 
#%% 
contingence = pd.crosstab(df.loc[df_quant.index, 'CLUSTER_ID'], 
df.loc[df_quant.index, 'NAF_A88']) 
chi2, p, dof, expected = chi2_contingency(contingence) 
print("Chi2 =", chi2, "p-value =", p) 
#%% Exportation de la base avec les clusters 
df.to_csv('C:/Users/IWTI4850/Desktop/votre_fichier_traite1.csv', index=False) 
df.to_excel('C:/Users/IWTI4850/Desktop/votre_fichier_traite1.xlsx', index=False)