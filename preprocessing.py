import pandas as pd

from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split

def preprocessing_data(link_data, options = None):

    if options is None:
        options = {'standarize' : True,
                   'standarize_output': False}

    # Read the file
    retards_df = pd.read_csv(link_data, sep=';')

    # Discard not used columns from the DataFrame
    data_to_delete = ["commentaire_annulation", "commentaires_retard_arrivee", 
                      "nb_annulation", "commentaire_retards_depart", "nb_train_retard_sup_15",
                      "retard_moyen_trains_retard_sup15","nb_train_retard_sup_30", 
                      "nb_train_retard_sup_60"]
    
    # Drop the specified columns from the DataFrame
    retards_df = retards_df.drop(columns=data_to_delete)

    # Calculate de total mean delay with existing columns
    retards_df['total_retard_mean'] = retards_df['retard_moyen_depart'] + retards_df['retard_moyen_arrivee']

    # Drop again
    retard_columns_to_delete = ["retard_moyen_depart","retard_moyen_tous_trains_depart", "retard_moyen_arrivee", "retard_moyen_tous_trains_arrivee"]
    retards_df = retards_df.drop(columns=retard_columns_to_delete)
    
    
    # Standarize to improve the model
    if options['standarize']:

    ## Not standarized
        scaler = StandardScaler()
        list_drop = ["date","service", "gare_depart","gare_arrivee"]
        if not options['standarize_output']:
            list_prct = ['prct_cause_externe', 'prct_cause_infra', 'prct_cause_gestion_trafic',
                          'prct_cause_materiel_roulant', 'prct_cause_gestion_gare',
                          'prct_cause_prise_en_charge_voyageurs']
            list_drop += list_prct
            retards_df[list_prct] = retards_df[list_prct]/100
        columns_to_standardize = retards_df.columns.drop(list_drop)
        retards_df[columns_to_standardize] = scaler.fit_transform(retards_df[columns_to_standardize])
    
    # Add moth as int 
    def get_month(row):
        date = int(row["date"][-2:])
        return date
    retards_df.insert(1,"mois",retards_df.apply(get_month, axis=1))

    # Create dummies of catergorical values of gare_depart, gare_arrivee and service
    dummies = pd.get_dummies(retards_df[["gare_depart","gare_arrivee","service"]])

    # Concatenate the original DataFrame with the dummy variables
    retards_w_dummies_df = pd.concat([retards_df, dummies], axis=1)
    retards_w_dummies_df = retards_w_dummies_df.drop(columns=["service","gare_depart","gare_arrivee"])

    # Split data frame with fiting and valudate
    fitting_df = retards_w_dummies_df[retards_w_dummies_df['date'] < '2023']
    validate_df = retards_w_dummies_df[retards_w_dummies_df['date'] >= '2023']

    fitting_df = fitting_df.drop("date",axis=1)
    validate_df = validate_df.drop("date",axis=1)
    
    data_to_mantain = ['prct_cause_externe', 'prct_cause_infra', 'prct_cause_gestion_trafic', 'prct_cause_materiel_roulant', 
                       'prct_cause_gestion_gare', 'prct_cause_prise_en_charge_voyageurs', 'total_retard_mean']

    df_Y = fitting_df[data_to_mantain]
    df_X = fitting_df.drop(columns=data_to_mantain)

    validate_df_X = validate_df[data_to_mantain]
    validate_df_Y = validate_df.drop(columns=data_to_mantain)

    train_df_X,test_df_X , train_df_Y,test_df_Y = train_test_split(df_X,df_Y, test_size=0.2)

    train_df_X = train_df_X.reset_index(drop=True)
    train_df_Y = train_df_Y.reset_index(drop=True)
    test_df_X = test_df_X.reset_index(drop=True)
    test_df_Y = test_df_Y.reset_index(drop=True)

    return  train_df_X, train_df_Y, test_df_X, test_df_Y, validate_df_X, validate_df_Y

if __name__ == "__main__":
    import numpy as np
    train_df_X, train_df_Y, test_df_X, test_df_Y, validate_df_X, validate_df_Y= preprocessing_data("data/regularite-mensuelle-tgv-aqst.csv")
    print(sum(train_df_Y.iloc[0,:-1]))
