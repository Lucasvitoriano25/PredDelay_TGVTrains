import pandas as pd

from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split

def preprocessing_data(link_data, options ={}):
    # 
    retards_df = pd.read_csv(link_data, sep=';')

    data_to_delete = ["commentaire_annulation", "commentaires_retard_arrivee", 
                      "nb_annulation", "commentaire_retards_depart", "nb_train_retard_sup_15",
                      "retard_moyen_trains_retard_sup15","nb_train_retard_sup_30", 
                      "nb_train_retard_sup_60"]
    
    # Drop the specified columns from the DataFrame
    retards_df = retards_df.drop(columns=data_to_delete)

    retards_df['total_retard_mean'] = retards_df['retard_moyen_depart'] + retards_df['retard_moyen_arrivee']
    retards_df['total_retard_alltrains_mean'] = retards_df['retard_moyen_tous_trains_depart'] + retards_df['retard_moyen_tous_trains_arrivee']

    retard_columns_to_delete = ["retard_moyen_depart","retard_moyen_tous_trains_depart", "retard_moyen_arrivee", "retard_moyen_tous_trains_arrivee"]
    retards_df = retards_df.drop(columns=retard_columns_to_delete)

    scaler = StandardScaler()
    columns_to_standardize = retards_df.columns.drop(["date","service", "gare_depart","gare_arrivee"])
    retards_df[columns_to_standardize] = scaler.fit_transform(retards_df[columns_to_standardize])

    dummies = pd.get_dummies(retards_df[["gare_depart","gare_arrivee","service"]])

    # Concatenate the original DataFrame with the dummy variables
    retards_w_dummies_df = pd.concat([retards_df, dummies], axis=1)

    retards_w_dummies_df = retards_w_dummies_df.drop(columns=["service","gare_depart","gare_arrivee"])

    fitting_df = retards_w_dummies_df[retards_w_dummies_df['date'] < '2023']
    validate_df = retards_w_dummies_df[retards_w_dummies_df['date'] >= '2023']

    fitting_df = fitting_df.drop("date",axis=1)
    validate_df = validate_df.drop("date",axis=1)

    data_to_mantain = ['prct_cause_externe', 'prct_cause_infra', 'prct_cause_gestion_trafic', 'prct_cause_materiel_roulant', 
                       'prct_cause_gestion_gare', 'prct_cause_prise_en_charge_voyageurs', 'total_retard_mean', 
                       'total_retard_alltrains_mean']

    df_Y = fitting_df[data_to_mantain]
    df_X = fitting_df.drop(columns=data_to_mantain)

    validate_df_X = validate_df[data_to_mantain]
    validate_df_Y = validate_df.drop(columns=data_to_mantain)

    train_df_X,test_df_X , train_df_Y,test_df_Y = train_test_split(df_X,df_Y, test_size=0.2)

    train_df_X = train_df_X.reset_index(drop=True)
    train_df_Y = train_df_Y.reset_index(drop=True)
    test_df_X = test_df_X.reset_index(drop=True)
    test_df_Y = test_df_Y.reset_index(drop=True)



    return  train_df_X, train_df_Y, test_df_X, test_df_Y 

if __name__ == "__main__":
    train_df_X, train_df_Y, test_df_X, test_df_Y  = preprocessing_data("data/regularite-mensuelle-tgv-aqst.csv")
    print(train_df_Y)