from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import LeaveOneOut
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import pandas as pd
import numpy as np


def calculate_metrics(metrics: np.array) -> dict:
    inner_dict = dict()
    inner_dict.update({'FP': (metrics.sum(axis=0) - np.diag(metrics)).astype(float)})
    inner_dict.update({'FN': (metrics.sum(axis=1) - np.diag(metrics)).astype(float)})
    inner_dict.update({'TP': (np.diag(metrics)).astype(float)})
    inner_dict.update({'TN': metrics.sum() - (inner_dict['FP'] + inner_dict['FN'] + inner_dict['TP'])})
    return inner_dict


def calculate_knn(n_neigh: int, split: bool = False) -> pd.DataFrame:
    iris = load_iris()
    iris_data = iris.data
    iris_target = iris.target

    if split:
        x_train, x_test, y_train,y_test = train_test_split(iris_data,iris_target, test_size=0.33)
    else:
        x_train, x_test = iris_data, iris_data
        y_train, y_test = iris_target, iris_target
    columns_df = ['method', 'class', 'accuracy', 'recall', 'specifity', 'precision', 'fmeasure', 'AUC', 'neigbours']
    final_df = pd.DataFrame(columns=columns_df)

    for i in range(1, n_neigh + 1):
        knn = KNeighborsClassifier(n_neighbors=i, metric='euclidean')
        knn.fit(x_train, y_train)

        print(f"Calculating for n:{i}")
        rows_data = calculate_quality(knn,x_test,y_test,'KNN')
        rows_data = rows_data.assign(neigbours = i)
        final_df = pd.concat([final_df, rows_data]).reset_index(drop=True)
    return final_df


def calculate_naive(split: bool = False):
    iris = load_iris()
    iris_data = iris.data
    iris_target = iris.target

    if split:
        x_train, x_test, y_train,y_test = train_test_split(iris_data,iris_target, test_size=0.33)
    else:
        x_train, x_test = iris_data, iris_data
        y_train, y_test = iris_target, iris_target

    gnb = GaussianNB()
    gnb.fit(x_train, y_train)

    return calculate_quality(gnb,x_test, y_test, 'GNB')


def calculate_svc(split: bool = False):
    iris = load_iris()
    iris_data = iris.data
    iris_target = iris.target

    if split:
        x_train, x_test, y_train,y_test = train_test_split(iris_data,iris_target, test_size=0.33)
    else:
        x_train, x_test = iris_data, iris_data
        y_train, y_test = iris_target, iris_target

    scaler = StandardScaler()
    scaler.fit(x_train)

    svc = SVC(gamma='auto', probability=True)
    svc.fit(x_train,y_train)

    return calculate_quality(svc,x_test,y_test,'SVC')


def calculate_logistic(split: bool = False):
    iris = load_iris()
    iris_data = iris.data
    iris_target = iris.target

    if split:
        x_train, x_test, y_train,y_test = train_test_split(iris_data,iris_target, test_size=0.33)
    else:
        x_train, x_test = iris_data, iris_data
        y_train, y_test = iris_target, iris_target

    log = LogisticRegression(solver='lbfgs', max_iter=500)
    log.fit(x_train,y_train)

    return calculate_quality(log,x_test,y_test,'LogisticRegression')


def calculate_quality(model, x_train, y_train, model_name):
    columns_df = ['method', 'class', 'accuracy', 'recall', 'specifity', 'precision', 'fmeasure', 'AUC']
    init_df = pd.DataFrame(columns=columns_df, data=[[None for j in range(len(columns_df))] for i in range(4)])
    init_df.loc[0:3, 'method'] = model_name
    init_df.loc[0, 'class'] = 'Iris-setosa'
    init_df.loc[1, 'class'] = 'Iris-versicolor'
    init_df.loc[2, 'class'] = 'Iris-virginica'
    init_df.loc[3, 'class'] = 'Overall weighted'

    y_pred = model.predict(x_train)
    cm = metrics.confusion_matrix(y_train, y_pred)
    print(f"CONFUSION MATRIX \n{cm}")
    r_dict = calculate_metrics(cm)

    TPR = r_dict['TP'] / (r_dict['TP'] + r_dict['FN'])  # czulosc
    TNR = r_dict['TN'] / (r_dict['FP'] + r_dict['TN'])  # swoistosc
    PPV = r_dict['TP'] / (r_dict['TP'] + r_dict['FP'])  # precyzja
    F_score = 2 * r_dict['TP'] / (2 * r_dict['TP'] + r_dict['FP'] + r_dict['FN'])  # fscore
    ACC = (r_dict['TP'] + r_dict['TN']) / (r_dict['TP'] + r_dict['FP'] + r_dict['FN'] + r_dict['TN'])

    init_df.loc[0, 'accuracy'] = ACC[0]
    init_df.loc[1, 'accuracy'] = ACC[1]
    init_df.loc[2, 'accuracy'] = ACC[2]
    init_df.loc[3, 'accuracy'] = metrics.accuracy_score(y_train, y_pred)

    init_df.loc[0, 'recall'] = TPR[0]
    init_df.loc[1, 'recall'] = TPR[1]
    init_df.loc[2, 'recall'] = TPR[2]
    init_df.loc[3, 'recall'] = metrics.recall_score(y_train, y_pred, average='weighted')

    init_df.loc[0, 'specifity'] = TNR[0]
    init_df.loc[1, 'specifity'] = TNR[1]
    init_df.loc[2, 'specifity'] = TNR[2]
    init_df.loc[3, 'specifity'] = np.average(TNR)

    init_df.loc[0, 'precision'] = PPV[0]
    init_df.loc[1, 'precision'] = PPV[1]
    init_df.loc[2, 'precision'] = PPV[2]
    init_df.loc[3, 'precision'] = metrics.precision_score(y_train, y_pred, average='weighted')

    init_df.loc[0, 'fmeasure'] = F_score[0]
    init_df.loc[1, 'fmeasure'] = F_score[1]
    init_df.loc[2, 'fmeasure'] = F_score[2]
    init_df.loc[3, 'fmeasure'] = metrics.f1_score(y_train, y_pred, average='weighted')

    auc_list = list()
    for i in range(3):
        fpr, tpr, treshold = metrics.roc_curve(y_train, model.predict_proba(x_train)[:, i], pos_label=i)
        auc_list.append(metrics.auc(fpr, tpr))

    init_df.loc[0, 'AUC'] = auc_list[0]
    init_df.loc[1, 'AUC'] = auc_list[1]
    init_df.loc[2, 'AUC'] = auc_list[2]
    init_df.loc[3, 'AUC'] = metrics.roc_auc_score(y_train, model.predict_proba(x_train), multi_class='ovr')

    return init_df


def cross_validate_all(leave_one_out: bool = False) -> pd.DataFrame:
    iris = load_iris()
    iris_data = iris.data
    iris_target = iris.target

    knn = KNeighborsClassifier(n_neighbors=1)
    log = LogisticRegression(solver='lbfgs', max_iter=500)
    gnb = GaussianNB()
    svc = SVC(gamma='auto', probability=True)

    columns = ['FitTime', 'ScoreTime', 'TestScore']
    models = ['KNN','SVM','LOGISTIC_REG','NAIVE_BAYES']
    mi = pd.MultiIndex.from_product([models, columns], names=['1','2']).tolist()
    mi.insert(0, ('', 'KNumbers'))
    mi.insert(1, ('', 'GroupNr'))
    mi_f = pd.MultiIndex.from_tuples(mi)

    if not leave_one_out:
        final_df = pd.DataFrame(columns=mi_f)
        for i in range(5,11):
            d_knn = cross_validate(knn,iris_data,iris_target,cv=i)
            d_log = cross_validate(log,iris_data,iris_target,cv=i)
            d_gnb = cross_validate(gnb,iris_data,iris_target,cv=i)
            d_svc = cross_validate(svc, iris_data, iris_target, cv=i)

            concat_d_knn = np.hstack(
                (d_knn['fit_time'][:,None], d_knn['score_time'][:,None],d_knn['test_score'][:,None]))
            concat_d_log = np.hstack(
                (d_log['fit_time'][:, None], d_log['score_time'][:, None], d_log['test_score'][:, None]))
            concat_d_gnb = np.hstack(
                (d_gnb['fit_time'][:, None], d_gnb['score_time'][:, None], d_gnb['test_score'][:, None]))
            concat_d_svc = np.hstack(
                (d_svc['fit_time'][:, None], d_svc['score_time'][:, None], d_svc['test_score'][:, None]))
            concat_models = np.hstack((concat_d_knn,concat_d_log,concat_d_gnb,concat_d_svc))
            group_nr_concat = np.array([j + 1 for j in range(i)])[:,None]
            k_number_concat = np.array([i for j in range(i)])[:,None]

            final_concat = np.hstack((k_number_concat,group_nr_concat,concat_models))
            inner_df = pd.DataFrame(columns=final_df.columns, data = final_concat)
            final_df = pd.concat([final_df,inner_df]).reset_index(drop=True)

    else:
        final_df = pd.DataFrame()
        loo = LeaveOneOut()
        nr_group = 1
        for train, test in loo.split(iris_data):
            x_train, x_test = iris_data[train], iris_data[test]
            y_train, y_test = iris_target[train], iris_target[test]

            knn = KNeighborsClassifier(n_neighbors=1)
            knn.fit(x_train,y_train)
            log = LogisticRegression(solver='lbfgs', max_iter=500)
            log.fit(x_train, y_train)
            gnb = GaussianNB()
            gnb.fit(x_train, y_train)
            svc = SVC(gamma='auto', probability=True)
            svc.fit(x_train, y_train)

            knn_data = calculate_quality(knn,iris_data,iris_target,'KNN')
            log_data = calculate_quality(log,iris_data,iris_target,'LOGISTIC')
            gnb_data = calculate_quality(gnb, iris_data, iris_target, 'NAIVE')
            svc_data = calculate_quality(svc, iris_data, iris_target, 'SVC')

            inner_df = pd.concat([knn_data, log_data, gnb_data, svc_data]).reset_index(drop=True)
            data_inner_concat = np.array(inner_df.values)
            concat_columns = list(inner_df.columns)
            concat_columns.insert(0, 'GroupNr')
            group_numbers_l = np.array([nr_group for i in range(len(data_inner_concat))])[:,None]

            if final_df.empty:
                final_df = pd.DataFrame(columns = concat_columns, data = np.hstack((group_numbers_l, data_inner_concat)))
            else:
                temp_df = pd.DataFrame(columns = concat_columns, data = np.hstack((group_numbers_l, data_inner_concat)))
                final_df = pd.concat([final_df, temp_df]).reset_index(drop=True)
            nr_group += 1

    return final_df

def leave_one_out_sec():
    iris = load_iris()
    iris_data = iris.data
    iris_target = iris.target

    knn = KNeighborsClassifier(n_neighbors=1)
    log = LogisticRegression(solver='lbfgs', max_iter=500)
    gnb = GaussianNB()
    svc = SVC(gamma='auto', probability=True)

    columns = ['FitTime', 'ScoreTime', 'TestScore']
    models = ['KNN', 'SVM', 'LOGISTIC_REG', 'NAIVE_BAYES']
    mi = pd.MultiIndex.from_product([models, columns], names=['1', '2']).tolist()
    mi.insert(0, ('', 'KNumbers'))
    mi.insert(1, ('', 'GroupNr'))
    mi_f = pd.MultiIndex.from_tuples(mi)

    final_df = pd.DataFrame(columns=mi_f)
    d_knn = cross_validate(knn, iris_data, iris_target, cv=50)
    d_log = cross_validate(log, iris_data, iris_target, cv=50)
    d_gnb = cross_validate(gnb, iris_data, iris_target, cv=50)
    d_svc = cross_validate(svc, iris_data, iris_target, cv=50)

    concat_d_knn = np.hstack(
        (d_knn['fit_time'][:, None], d_knn['score_time'][:, None], d_knn['test_score'][:, None]))
    concat_d_log = np.hstack(
        (d_log['fit_time'][:, None], d_log['score_time'][:, None], d_log['test_score'][:, None]))
    concat_d_gnb = np.hstack(
        (d_gnb['fit_time'][:, None], d_gnb['score_time'][:, None], d_gnb['test_score'][:, None]))
    concat_d_svc = np.hstack(
        (d_svc['fit_time'][:, None], d_svc['score_time'][:, None], d_svc['test_score'][:, None]))
    concat_models = np.hstack((concat_d_knn, concat_d_log, concat_d_gnb, concat_d_svc))
    group_nr_concat = np.array([j + 1 for j in range(50)])[:, None]
    k_number_concat = np.array([50 for j in range(50)])[:, None]

    final_concat = np.hstack((k_number_concat, group_nr_concat, concat_models))
    inner_df = pd.DataFrame(columns=final_df.columns, data=final_concat)
    final_df = pd.concat([final_df, inner_df]).reset_index(drop=True)

    return final_df


def process_all_data():
    knn_data = calculate_knn(30)
    log_data = calculate_logistic()
    bayess_data = calculate_naive()
    svc_data = calculate_svc()

    split_knn_data = calculate_knn(30,True)
    split_log_data = calculate_logistic(True)
    split_bayess_data = calculate_naive(True)
    split_svc_data = calculate_svc(True)

    cross_validate_data = cross_validate_all()
    leave_one_out_data = cross_validate_all(leave_one_out=True)

    with pd.ExcelWriter('results_ver2.xlsx') as writer:
        knn_data.to_excel(writer, sheet_name='knn_data', index=False)
        log_data.to_excel(writer, sheet_name='log_data', index=False)
        bayess_data.to_excel(writer, sheet_name='bayess_data', index=False)
        svc_data.to_excel(writer, sheet_name='svc_data', index=False)

        split_knn_data.to_excel(writer, sheet_name='split_knn_data', index=False)
        split_log_data.to_excel(writer, sheet_name='split_log_data', index=False)
        split_bayess_data.to_excel(writer, sheet_name='split_bayess_data', index=False)
        split_svc_data.to_excel(writer, sheet_name='split_svc_data', index=False)

        cross_validate_data.to_excel(writer, sheet_name='cross_validate_data')
        leave_one_out_data.to_excel(writer, sheet_name='leave_one_out_data',index=False)


print(calculate_knn(16))




