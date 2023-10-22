# import dim_reduction e fazer funções pra pegar x features do pca e melhor modelo do forward/backward

# fazer pca separado pra dummy e n dummy (no outro arquivo)

import preprocessing as pp
import numpy as np
import pandas as pd
import time
from sklearn.decomposition import PCA

# train_df_X, train_df_Y, test_df_X, test_df_Y  = pp.preprocessing_data("data/regularite-mensuelle-tgv-aqst.csv")

def get_pca_features(train_df_X, explained_variance_ratio=100):
    
    pca = PCA()
    X_reduced_train = pca.fit_transform(train_df_X)

    var_explained = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
    print(var_explained)

    number_of_features = np.argmax(var_explained >= explained_variance_ratio)

    if explained_variance_ratio==100:
        return X_reduced_train

    return X_reduced_train[:, :number_of_features]


def best_forward_feature_selection():
    # Best features as obtained in the dim_reduction.ipynb analysis
    best_features = ['duree_moyenne', 'gare_depart_PARIS LYON', 'service_International', 'gare_arrivee_PARIS MONTPARNASSE', 'nb_train_retard_arrivee', 'nb_train_prevu', 'nb_train_depart_retard', 'gare_arrivee_PARIS NORD', 'gare_depart_BARCELONA', 'gare_depart_PARIS EST', 'service_National', 'gare_depart_PARIS MONTPARNASSE', 'gare_depart_LYON PART DIEU', 'gare_depart_QUIMPER', 'gare_arrivee_PARIS EST', 'gare_depart_ITALIE', 'gare_depart_MARNE LA VALLEE', 'gare_depart_LILLE', 'gare_depart_LE CREUSOT MONTCEAU MONTCHANIN', 'gare_arrivee_GENEVE', 'gare_depart_TOURS', 'gare_arrivee_TOURCOING', 'gare_depart_MADRID', 'gare_arrivee_BARCELONA', 'gare_depart_ANNECY', 'gare_depart_NIMES', 'gare_depart_VALENCE ALIXAN TGV', 'gare_depart_CHAMBERY CHALLES LES EAUX', 'gare_depart_PERPIGNAN', 'gare_depart_MACON LOCHE', 'gare_arrivee_TOURS', 'gare_arrivee_DOUAI', 'gare_arrivee_ANGOULEME', 'gare_arrivee_LAVAL', 'gare_arrivee_ST MALO', 'gare_arrivee_ST PIERRE DES CORPS', 'gare_depart_BORDEAUX ST JEAN', 'gare_arrivee_VANNES', 'gare_arrivee_FRANCFORT', 'gare_depart_NICE VILLE', 'gare_depart_TOURCOING', 'gare_arrivee_LE CREUSOT MONTCEAU MONTCHANIN', 'gare_arrivee_BORDEAUX ST JEAN', 'gare_depart_LAVAL', 'gare_arrivee_LILLE', 'gare_depart_STUTTGART', 'gare_depart_GENEVE', 'gare_arrivee_LAUSANNE', 'gare_arrivee_MACON LOCHE', 'gare_arrivee_BREST', 'gare_depart_TOULOUSE MATABIAU', 'gare_arrivee_CHAMBERY CHALLES LES EAUX', 'gare_depart_AVIGNON TGV', 'gare_depart_TOULON', 'gare_depart_AIX EN PROVENCE TGV', 'gare_depart_STRASBOURG', 'gare_arrivee_NICE VILLE', 'gare_depart_MARSEILLE ST CHARLES', 'gare_arrivee_LE MANS', 'gare_depart_NANTES', 'gare_depart_MONTPELLIER', 'gare_depart_ANGOULEME', 'gare_arrivee_SAINT ETIENNE CHATEAUCREUX', 'gare_depart_LE MANS', 'gare_arrivee_LYON PART DIEU', 'gare_arrivee_QUIMPER', 'gare_arrivee_DIJON VILLE', 'gare_arrivee_AIX EN PROVENCE TGV', 'gare_arrivee_GRENOBLE', 'gare_arrivee_ANNECY', 'gare_arrivee_VALENCE ALIXAN TGV', 'gare_depart_LA ROCHELLE VILLE', 'gare_depart_ST PIERRE DES CORPS', 'gare_depart_ST MALO', 'gare_depart_POITIERS', 'gare_depart_VANNES', 'gare_arrivee_LA ROCHELLE VILLE', 'gare_depart_METZ', 'gare_arrivee_BESANCON FRANCHE COMTE TGV', 'gare_arrivee_NIMES', 'gare_depart_NANCY', 'gare_arrivee_TOULOUSE MATABIAU', 'gare_arrivee_ANGERS SAINT LAUD', 'gare_arrivee_NANTES', 'gare_arrivee_ARRAS', 'gare_depart_PARIS NORD', 'gare_arrivee_PARIS LYON', 'gare_arrivee_STRASBOURG', 'gare_depart_BREST', 'gare_arrivee_DUNKERQUE', 'gare_depart_ARRAS', 'gare_arrivee_PERPIGNAN', 'gare_arrivee_TOULON', 'gare_depart_LAUSANNE', 'gare_arrivee_METZ', 'gare_arrivee_MARSEILLE ST CHARLES', 'gare_arrivee_RENNES']
    return best_features


train_df_X, train_df_Y, test_df_X, test_df_Y, validate_df_X, validate_df_Y  = pp.preprocessing_data("data/regularite-mensuelle-tgv-aqst.csv")

