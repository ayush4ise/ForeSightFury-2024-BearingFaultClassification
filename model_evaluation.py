import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.base import ClassifierMixin

def evaluate(features: pd.DataFrame, model: ClassifierMixin):
    """
    This function evaluates the model for the given features

    Args:
        features: features DataFrame to be evaluated
        model: model to be evaluated
    """
    df = features

    # separate the data by fault and add normal data to each fault
    fault7 = df[df['defect_type'] == 0.007]
    fault14 = df[df['defect_type'] == 0.014]
    fault21 = df[df['defect_type'] == 0.021]
    normal = df[df['defect_type'] == 0]

    fault7 = pd.concat([fault7, normal])
    fault14 = pd.concat([fault14, normal])
    fault21 = pd.concat([fault21, normal])

    faults = {7: fault7, 14: fault14, 21: fault21}

    # print accuracy table for each fault
    for fault_size in faults.keys():
        faultdf = faults[fault_size]

        # split the data into train and test sets for each hp
        X_0 = faultdf[faultdf['hp'] == 0].drop(['hp', 'target', 'defect_type'], axis=1)
        X_1 = faultdf[faultdf['hp'] == 1].drop(['hp', 'target', 'defect_type'], axis=1)
        y_0 = faultdf[faultdf['hp'] == 0]['target']
        y_1 = faultdf[faultdf['hp'] == 1]['target']

        X_train_0, X_test_0, y_train_0, y_test_0 = train_test_split(X_0, y_0, test_size=0.2, random_state=42)
        X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_1, y_1, test_size=0.2, random_state=42)

        # train the model
        model_0 = model # model trained on 0hp
        model_0.fit(X_train_0, y_train_0)
        model_1 = model # model trained on 1hp
        model_1.fit(X_train_1, y_train_1)

        # predict the test set
        # Training load- 0hp, Testing load- 0hp
        y_pred_00 = model_0.predict(X_test_0)
        # Training load- 0hp, Testing load- 1hp
        y_pred_01 = model_0.predict(X_1)

        # Training load- 1hp, Testing load- 0hp
        y_pred_10 = model_1.predict(X_0)
        # Training load- 1hp, Testing load- 1hp
        y_pred_11 = model_1.predict(X_test_1)

        # accuracy table
        accuracy_table = pd.DataFrame(columns=['Training Load', 'Testing Load', 'Accuracy'])
        accuracy_table['Training Load'] = [0, 0, 1, 1]
        accuracy_table['Testing Load'] = [0, 1, 0, 1]
        accuracy_table['Accuracy'] = [accuracy_score(y_test_0, y_pred_00),
                                    accuracy_score(y_1, y_pred_01),
                                    accuracy_score(y_0, y_pred_10),
                                    accuracy_score(y_test_1, y_pred_11)]

        print(f'Accuracy Table for fault {fault_size}')
        print(accuracy_table) 

# # getting the data for the model evaluation
# df = pd.read_csv('mix_features.csv')

# # separate the data by fault and add normal data to each fault
# fault7 = df[df['defect_type'] == 0.007]
# fault14 = df[df['defect_type'] == 0.014]
# fault21 = df[df['defect_type'] == 0.021]
# normal = df[df['defect_type'] == 0]

# fault7 = pd.concat([fault7, normal])
# fault14 = pd.concat([fault14, normal])
# fault21 = pd.concat([fault21, normal])

# faults = {7: fault7, 14: fault14, 21: fault21}

# fault_size = 7
# faultdf = faults[fault_size]

# # split the data into train and test sets for each hp
# X_0 = faultdf[faultdf['hp'] == 0].drop(['hp', 'target', 'defect_type'], axis=1)
# X_1 = faultdf[faultdf['hp'] == 1].drop(['hp', 'target', 'defect_type'], axis=1)
# y_0 = faultdf[faultdf['hp'] == 0]['target']
# y_1 = faultdf[faultdf['hp'] == 1]['target']

# X_train_0, X_test_0, y_train_0, y_test_0 = train_test_split(X_0, y_0, test_size=0.2, random_state=42)
# X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_1, y_1, test_size=0.2, random_state=42)

# # train the model
# lda_0 = LDA()
# lda_0.fit(X_train_0, y_train_0)
# lda_1 = LDA()
# lda_1.fit(X_train_1, y_train_1)

# # predict the test set
# # Training load- 0hp, Testing load- 0hp
# y_pred_00 = lda_0.predict(X_test_0)
# # Training load- 0hp, Testing load- 1hp
# y_pred_01 = lda_0.predict(X_1)

# # Training load- 1hp, Testing load- 0hp
# y_pred_10 = lda_1.predict(X_0)
# # Training load- 1hp, Testing load- 1hp
# y_pred_11 = lda_1.predict(X_test_1)

# # accuracy table
# accuracy_table = pd.DataFrame(columns=['Training Load', 'Testing Load', 'Accuracy'])
# accuracy_table['Training Load'] = [0, 0, 1, 1]
# accuracy_table['Testing Load'] = [0, 1, 0, 1]
# accuracy_table['Accuracy'] = [accuracy_score(y_test_0, y_pred_00),
#                               accuracy_score(y_1, y_pred_01),
#                               accuracy_score(y_0, y_pred_10),
#                               accuracy_score(y_test_1, y_pred_11)]

# print(f'Accuracy Table for fault {fault_size}')
# print(accuracy_table)