# TGV Train Delay Prediction

## Overview

In the dynamic world of transportation, the timeliness of high-speed trains is crucial for ensuring efficient and dependable service. Punctuality not only reflects service quality but also the railway's ability to manage and mitigate delays. In the context of the French high-speed rail system (Train à Grande Vitesse - TGV), where precision is paramount, predicting train arrival delays and understanding their root causes are of great importance.

This GitHub repository hosts a project focused on predicting and understanding TGV train delays, particularly those operated by the Société Nationale des Chemins de fer Français (SNCF), the primary railway company in France. We utilize various machine learning and deep learning techniques to achieve these objectives.

## Tools and Libraries

### XGBoost Model

For the XGBoost model, we employ the following libraries:

- [XGBoost](https://xgboost.readthedocs.io/en/latest/)
- [Pickle](https://docs.python.org/3/library/pickle.html)
- Other standard Python libraries for data manipulation and visualization.

### Neural Network Model

For the neural network model, we rely on the following libraries:

- [TensorFlow](https://www.tensorflow.org/)
- [TensorFlow Decision Forests](https://www.tensorflow.org/decision_forests)
- [Scikit-learn](https://scikit-learn.org/stable/)
- Various libraries for data preprocessing and analysis.

### Random Forest Model

To implement the Random Forest model, we utilize the following libraries:

- [LightGBM](https://lightgbm.readthedocs.io/en/latest/)
- [PyCaret](https://pycaret.org/)
- [Scikit-learn](https://scikit-learn.org/stable/)
- Other standard Python libraries.

### Dimensionality Reduction

For dimensionality reduction techniques, we rely on:

- [Statsmodels](https://www.statsmodels.org/stable/index.html)
- Standard Python libraries for data manipulation.


## Model Objectives and Comparison
To have a better understanding of the predictions performance, each model was split into two objectives. One objective predicts the total mean delay (total_retard_mean), while the other predicts the percentage of the cause of the delay, forecasting 7 features related to it. We explored the models of Neural Network, XGBoost, and Random Forest, as described in the following section.

Comparing the Models
This table compares the performance of three different machine learning models in terms of $R^2$ and root mean square error for three different models (predicting Mean delay, predicting percentage, and predicting both models). The scores correspond to the test dataset and provide valuable insights into the models' performance.

Overall, the NN Model performed the best in predicting mean delay, the XGBoost Model performed the best in predicting PRCT, and the Random Forest performed the best in predicting both mean and PRCT.

### Results: 
![model_prct](https://github.com/Lucasvitoriano25/PredDelay_TGVTrains/assets/52925699/443638a1-172c-485a-a16c-3d1853b952f5)
![model_mean](https://github.com/Lucasvitoriano25/PredDelay_TGVTrains/assets/52925699/28bedd13-4dc9-4967-b233-ff54f5c24d43)
![model_both](https://github.com/Lucasvitoriano25/PredDelay_TGVTrains/assets/52925699/760fd845-ab23-45c3-a248-36869b0c996e)

![Screenshot from 2023-10-23 14-38-11](https://github.com/Lucasvitoriano25/PredDelay_TGVTrains/assets/52925699/19d0e360-45f7-44c8-9a34-c45ca3305d53)
![Screenshot from 2023-10-23 14-38-06](https://github.com/Lucasvitoriano25/PredDelay_TGVTrains/assets/52925699/de3798d1-89d4-47af-8d89-f3ae1eaf1f89)




| Approach       | Experiment        | R2 Score | Root Mean Squared Error |
|----------------|-------------------|----------|--------------------------|
| NN Model       | Predicting Mean   | 0.4287   | 0.7642                   |
|                | Predicting PRCT   | 0.1519   | 0.1259                   |
|                | Predicting Both   | 0.2014   | 0.3048                   |
| XGBoost Model  | Predicting Mean   | -0.7920  | 0.8380                   |
|                | Predicting PRCT   | 0.1548   | 0.1233                   |
|                | Predicting Both   | 0.1324   | 0.1193                   |
| Random Forest  | Predicting Mean   | 0.3938   | 0.7879                   |
|                | Predicting PRCT   | 0.1991   | 0.1181                   |
|                | Predicting Both   | 0.2143   | 0.2177                   |

## Usage

- Clone this repository to your local machine:

```bash
git clone https://github.com/your-username/tgv-train-delay-prediction.git´´´ 

