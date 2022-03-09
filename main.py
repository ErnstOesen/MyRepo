
from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import seaborn as sns

from sklearn.cluster import KMeans


# helper
def check_nan(string):
    """
    checks for empty string
    """
    return string != string


def merge(m1, m2):
    res = pd.merge(m1, m2, on=['id'])
    return res



# read & write
def read_min_out(path, top, time_metric, priority): # time,criterion
    """
    reads "min_out"-file, creates feature vector for each file and final feature matrix

    :param path: path to min_out files
    :param top: number of extracted flows
    :param time_metric: ground truth values for Real User Monitoring Speed Index (RUMSI) / Page Load Time (PLT)
    :param priority: sorts flows by size/frequency
    :return: feature matrix for model training
    """

    import os
    directory = os.fsencode(path)

    matrix = list()
    feature_names = get_feature_names(top)
    id_names = list()

    # iterate over "min_out"-files
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith("min_out.log"):
            if os.stat(path + filename).st_size < 2:
                continue
            else:
                df = pd.read_csv(r"path" + filename,
                    error_bad_lines=False, sep=';', na_values=0, index_col=False)
                if len(df.index) < 3:
                    continue

                # save file name as "id"
                id = filename[:len(filename)-14]
                id_names.append(id)

                # creates feature vector from "min_out"-file
                vector = create_vector_min_out(df, id, top, time_metric, priority)
                if vector is None:
                    continue

                # appends feature vector to dataframe
                matrix.append(vector)


    matrix = pd.DataFrame(matrix)
    matrix.columns = feature_names
    matrix.to_csv(index=False)

    return matrix


def read_rumsi(path, dir):
    """
    reads ground truth value for RUMSI

    :param path: path to directories
    :param dir: directory with RUMSI values
    :return: column of RUMSI values for given feature matrix
    """

    import math
    gt = pd.read_csv(r"D:/Uni Ernst/" + path + dir + "summary.csv",
                     index_col=False, sep=',', na_values=0,compression='zip')

    rumsi_map = dict()
    size = gt.index.__len__()

    for row in range (size):
        id = gt["id"].iloc[row]
        rumsi = gt["RUMSI"].iloc[row]
        if math.isnan(rumsi):
            continue
        rumsi_map[id] = rumsi

    rumsi_df = pd.DataFrame()
    rumsi_df["id"] = rumsi_map.keys()
    rumsi_df["rumsi"] = rumsi_map.values()
    return rumsi_df


def read_plt(path,dir):
    """
    reads ground truth value for Page Load Time (PLT)

    :param path: path to directories
    :param dir: directory with PLT values
    :return: column of PLT values for given feature matrix
    """

    import math
    gt = pd.read_csv(r"D:/Uni Ernst/" + path + dir + "summary.csv",
                     index_col=False, sep=',', na_values=0)

    import math

    plt_map = dict()
    size = gt.index.__len__()

    for row in range (size):
        id = gt["id"].iloc[row]
        plt = gt["aatf"].iloc[row]
        if check_nan(plt):
            continue
        segments = plt.split(",")[2]
        value = segments.split(":")
        plt = value[1]
        plt_map[id] = float(plt)

    plt_df = pd.DataFrame()
    plt_df["id"] = plt_map.keys()
    plt_df["plt"] = plt_map.values()
    return plt_df


def read_df_from_file(filename):
    """
    reads existing dataframe by filename

    :param filename:
    :return: dataframe
    """
    dataframe = pd.read_csv(r"C:/Users/RNEE/PycharmProjects/firstProgram/" + filename,
                          index_col=False, sep=',', na_values=0)  # 4630
    dataframe = dataframe.fillna(0)
    return dataframe


def write_df_to_csv(df, name):
    """
    creates csv file of dataframe
    """
    df.to_csv(r"C:/Users/RNEE/PycharmProjects/firstProgram/" + name,index=False)


def get_feature_names(top):
    """
    creates name of feature vector for all flows of file

    :param top: number of flows
    :return: feature name of vector
    """
    feature_names = list()
    feature_names.append("id")
    for i in range(top):
        feature_names.append("sum" + str(i))
        feature_names.append("count" + str(i))
        feature_names.append("mean" + str(i))
        feature_names.append("std" + str(i))
        feature_names.append("cv" + str(i))
        feature_names.append("kurt" + str(i))
        feature_names.append("skew" + str(i))
        feature_names.append("min" + str(i))
        feature_names.append("max" + str(i))
        feature_names.append("median" + str(i))
    return feature_names



# preprocessing
def pre_processing_by_perc(matrix, percentile):
    """
    eliminate outlier by deleting rows of dataframe for given percentile

    :param matrix: feature matrix
    :param percentile: threshold for RUMSI / PLT
    :return: modified matrix
    """

    length = len(matrix.index)
    errors = list()
    for column in matrix:
        if column == 'id':
            continue
        quant = matrix[column].astype('float').quantile(percentile)
        for i in range(length):
            if (matrix[column].iloc[i] > quant):
                errors.append(i)
                matrix[column].iloc[i] = np.nan
    matrix = matrix.dropna(axis=0)
    return matrix


def pre_processing_by_gt(matrix, gt, value):
    """
    eliminate outlier by deleting rows of dataframe for given threshold
    exclusive operation for RUMSI / PLT

    :param matrix: feature matrix
    :param gt: RUMSI / PLT
    :param value: threshold for RUMSI / PLT
    :return: modified matrix
    """

    length = len(matrix.index)
    errors = list()
    for i in range(length):
        if (matrix[gt].iloc[i] > value):
            errors.append(i)
            matrix[gt].iloc[i] = np.nan
    matrix = matrix.dropna(axis=0)
    return matrix


def create_heatmap(feature_set):
    """
    computes spearman correlation of feature set with RUMSI / PLT
    creates visual heatmap of correlation

    :param feature_set:
    :return: correlation vectors for RUMSI / PLT
    """

    correlation_map = feature_set.corr('spearman')

    plt.figure(figsize=(10, 6))
    sns.set(font_scale=.7)
    hmap = sns.heatmap(correlation_map, xticklabels=True, yticklabels=True)
    hmap.set_xticklabels(hmap.get_xticklabels(), rotation=75)
    plt.imshow(correlation_map, cmap='hot', interpolation='nearest', aspect='auto', extent=[0, 100, 0, 1])
    plt.show()

    correlations_rumsi = correlation_map["rumsi"]
    correlations_rumsi = correlations_rumsi.sort_values(ascending=False)

    correlation_plt = correlation_map["plt"]
    correlation_plt = correlation_plt.sort_values(ascending=False)

    return correlations_rumsi, correlation_plt


def k_means(df, type, k):
    """
    clustering of ground truth values (RT/IAT)

    :param gt: given dataframe
    :param type: RT / IAT
    :param k: number of clusters
    :return: modified dataframe with additional cluster feature
    """

    values = df[type].values.reshape(len(df.index), 1)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(values)

    # add new column as cluster feature
    df["cluster_" + type] = kmeans.labels_.astype(float)
    df["cluster_" + type] = df["cluster_" + type] * 1000
    return df



# create feature sets
def compute_distribution_statistics(arr):
    """
    computes statistics of numerical array

    :param arr: list of features
    :return: single feature vector for list of computed server response times (RT) / inter arrival times (IAT)
    """
    import numpy as np
    from scipy.stats import kurtosis, skew

    statistics = dict()
    if (len(arr) != 0):
        # cv = coefficient of deviation, skew = skewnes, kurt = kurtosis
        statistics["sum"] = sum(arr)
        statistics["count"] = len(arr)
        statistics["mean"] = np.mean(arr)
        statistics["std"] = np.std(arr)
        if (statistics["mean"] != 0):
            statistics["cv"] = statistics["std"] / statistics["mean"]
        else:
            statistics["cv"] = np.nan
        statistics["kurt"] = kurtosis(arr, fisher=False)
        statistics["skew"] = skew(arr)
        statistics["min"] = np.min(arr)
        statistics["max"] = np.max(arr)
        statistics["median"] = np.percentile(arr, 50)
    else:
        statistics["sum"] = np.nan
        statistics["count"] = np.nan
        statistics["mean"] = np.nan
        statistics["std"] = np.nan
        statistics["cv"] = np.nan
        statistics["kurt"] = np.nan
        statistics["skew"] = np.nan
        statistics["min"] = np.nan
        statistics["max"] = np.nan
        statistics["median"] = np.nan
    return statistics


def create_vector_min_out(df, filename, top, type, criterion):
    """
    computes server response times (RT) and inter arrival times (IAT) for each flow
    creates feature vector for set of RTs and IATs
    adds up most important flows to one feature vector

    :param df: csv file
    :param filename: index names of dataframe
    :param top:  number of most important flows
    :param type: features of RT / IAT
    :param criterion: flows sorted by priority
    :return: feature vector of top flows
    """

    feature_list = list()
    feature_list.append(filename)

    response_times = dict()   # stores rts of flows
    arrival_times = dict()    # stores arrival times of incoming package for each flow
    id_rt_sizes = dict()      # sums total time of rts

    id_sn_map = dict()        # finds flows
    flow_sizes = dict()       # number of transported bytes in flow


    # computes server response times (RT) and inter arrival times (IAT) for all flows
    for i in range(len(df.index)):

        row = df.iloc[i]
        time = row["Time"]
        src_ip = row["SrcIP"]
        dst_ip = row["DstIP"]
        src_port = row["SrcPort"]
        dst_port = row["DstPort"]
        size = row["Size"]
        details = str(row["Details"])
        details_array = details.split(",")

        if len(details_array) != 4 or details.startswith("DNS"):
            continue

        details_array = details.split(",")

        sn = details_array[0]
        if(sn.startswith("SN")):
            sn = sn.split(":")[1]

        an = details_array[1]
        if (an.startswith("AN")):
            an = an.split(":")[1]

        src_key = src_ip + "," + str(src_port) + "," + sn
        dst_key = dst_ip + "," + str(dst_port) + "," + an

        src_flow = src_ip + "," + str(src_port) + "," + dst_ip + "," + str(dst_port)
        dst_flow = dst_ip + "," + str(dst_port) + "," + src_ip + "," + str(src_port)


        value = str(time) + "," + str(i)
        id_sn_map[src_key] = [value]

        if check_nan(size):
            size = 0

        # src known, add size of bytes
        if src_flow in flow_sizes:
            curr_size = flow_sizes[src_flow]
            sum = int(curr_size) + int(size)
            flow_sizes[src_flow] = sum

        # dst known, add size of bytes
        else:
            if dst_flow in flow_sizes:
                curr_size = flow_sizes[dst_flow]
                sum = int(curr_size) + int(size)
                flow_sizes[dst_flow] = sum
            else:
                flow_sizes[src_flow] = 0


        # transaction is response, calculate RT
        if dst_key in id_sn_map:
            res_row = id_sn_map[dst_key]
            old_time = res_row[0].split(",")[0]
            old_row = res_row[0].split(",")[1]

            for i in range (6):
                time = time * 10
            time = int(time)
            old_time = float(old_time)
            old_time = old_time * 10
            for i in range(5):
                old_time = old_time * 10
            old_time = int(old_time)
            value = (time - old_time) / 1000

            if dst_flow not in response_times:
                response_times[dst_flow] = list()
                arrival_times[dst_flow] = list()

            response_times[dst_flow].append(value)
            arrival_times[dst_flow].append(time)

            # update frequency of flow
            if dst_flow in id_rt_sizes:
                curr = id_rt_sizes[dst_flow]
                id_rt_sizes[dst_flow] = curr + 1
            else:
                id_rt_sizes[dst_flow] = 1


        # transaction is request
        else:
            if src_flow in id_rt_sizes:
                curr = id_rt_sizes[src_flow]
                id_rt_sizes[src_flow] = curr + 1
            else:
                id_rt_sizes[src_flow] = 1
            if src_flow not in response_times:
                response_times[src_flow] = list()
                arrival_times[src_flow] = list()

    errors = list()
    final_response_times = dict()
    final_arrival_times = dict()


    # merge both directions of flows, eliminate duplicates
    for key in response_times.keys():

        rt_1 = response_times[key]

        ipsrc = key.split(",")[0]
        portsrc = key.split(",")[1]
        ipdst = key.split(",")[2]
        portdst = key.split(",")[3]
        reverse_key = ipdst + "," + portdst + "," + ipsrc + "," + portsrc
        if reverse_key in final_response_times.keys():
            continue
        elif reverse_key in response_times.keys():
            rt_2 = response_times[reverse_key]
            rt_1 = rt_1 + rt_2

        final_response_times[key] = rt_1

    # merge both directions of flows, eliminate duplicates
    for key in arrival_times.keys():

        iats_1 = arrival_times[key]

        ipsrc = key.split(",")[0]
        portsrc = key.split(",")[1]
        ipdst = key.split(",")[2]
        portdst = key.split(",")[3]
        reverse_key = ipdst + "," + portdst + "," + ipsrc + "," + portsrc
        if reverse_key in final_arrival_times.keys():
            continue
        elif reverse_key in arrival_times.keys():
            iats_2 = arrival_times[reverse_key]
            iats_1 = iats_1 + iats_2
            iats_1.sort()

        final_arrival_times[key] = iats_1

    # find empty sets of RT and IAT
    for key in final_response_times.keys():
        if final_response_times.get(key).__len__() == 0:
            errors.append(key)

    # delete empty sets of RT and IAT
    for key in errors:
        del final_response_times[key]
        del arrival_times[key]


    # prioritize flows by "length" = number of transactions (frequency), "size" = number of bytes
    if criterion == "length":
        # rt_length
        sorted_sizes = dict(sorted(id_rt_sizes.items(), key=lambda item: item[1]))
    elif criterion == "size":
        # size
        sorted_sizes = dict(sorted(flow_sizes.items(), key=lambda item: item[1]))

    keys = list(sorted_sizes.keys())
    keys.reverse()
    length = len(keys)
    counter = 0


    # computes set of RT / IAT for most important flows by "criterion"
    for i in range(length):
        if counter == top:
            break
        curr_key = keys[i]
        ipsrc = curr_key.split(",")[0]
        portsrc = curr_key.split(",")[1]
        ipdst = curr_key.split(",")[2]
        portdst = curr_key.split(",")[3]
        reverse_key = ipdst + "," + portdst + "," + ipsrc + "," + portsrc

        # compute feature vector of response times (RT)
        if type == "rt":
            if curr_key in final_response_times.keys():
                rts = np.asarray(final_response_times[curr_key])
                if len(rts) < 10:
                    continue
                counter += 1
            elif reverse_key in final_response_times.keys():
                rts = np.asarray(final_response_times[reverse_key])
                if len(rts) < 10:
                    continue
                counter += 1
            else:
                continue

            # single feature vector by RT
            features_rts = compute_distribution_statistics(rts)

            # append feature vector to flow vector
            for f in features_rts.values():
                feature_list.append(f)



        # compute feature vector of inter arrival times (IAT)
        elif type == "iat":
            if curr_key in final_arrival_times.keys():
                iats = np.array(get_inter_arrival_times(final_arrival_times[curr_key]))

                if len(iats) < 10:
                    continue
                counter += 1
            elif reverse_key in final_arrival_times.keys():
                iats = np.array(get_inter_arrival_times(final_arrival_times[reverse_key]))
                if len(iats) < 10:
                    continue
                counter += 1
            else:
                continue

            # single feature vector by IAT
            features_iats = compute_distribution_statistics(iats)

            # append feature vector to flow vector
            for f in features_iats.values():
                feature_list.append(f)


    return feature_list


def get_inter_arrival_times(arrival_times):
    """
    computes set of inter arrival times (IATs) by given timestamps

    :param arrival_times: timestamps
    :return: list of IATs
    """
    times = list(arrival_times)
    inter_arrival_times = list()
    for i in range(len(arrival_times)-1):
        time_1 = times[i]
        time_2 = times[i+1]
        time = time_2 - time_1
        time = time / 1000
        inter_arrival_times.append(time)
    return inter_arrival_times



# model training and evaluation
def multi_linear_regressor(matrix, ground_truth):
    """
    training of multi linear regressor and approximation of ground truth
    evaluation by error metrics

    :param matrix: feature matrix
    :param ground_truth: approximated RUMSI / PLT
    :return: error metrics mean squared error (MSE), mean absolute error (MAE), mean absolute percentage error (MAPE),
             median absolute error (MDAE)
    """

    clf = LinearRegression()

    # df = training set, gt = ground truth
    matrix = matrix.dropna(axis=0)
    gt = matrix[ground_truth]
    del matrix[ground_truth]
    df = matrix

    # Split: 90% training, 10% test
    X_train, X_test, y_train, y_test = train_test_split(df, gt, test_size=0.1)

    # initialize scaler
    X_scaler = MinMaxScaler(feature_range=(0.01, 1))
    y_scaler = MinMaxScaler(feature_range=(0.01, 1))

    # shaping
    y_train = np.array(y_train)
    y_train = y_train.reshape(len(y_train), 1)

    # model training
    X_scaler = X_scaler.fit(X_train)
    y_scaler = y_scaler.fit(y_train)

    # scaling
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)
    y_train = y_scaler.transform(y_train)

    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="neg_mean_squared_error")

    clf = clf.fit(X_train, y_train)

    # reverse scaling
    y_pred = clf.predict(X_test)
    y_pred = np.array(y_pred)
    y_pred = y_pred.reshape(len(y_pred), 1)
    y_pred = y_scaler.inverse_transform(y_pred)

    # evaluation by error metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mdae = median_absolute_error(y_test, y_pred)

    return mse, mae, mape, mdae


def decision_tree_regressor(matrix, ground_truth):
    """
        training of decision tree regressor and approximation of ground truth
        evaluation by error metrics

        :param matrix: feature matrix
        :param ground_truth: approximated RUMSI / PLT
        :return: error metrics mean squared error (MSE), mean absolute error (MAE), mean absolute percentage error (MAPE),
                 median absolute error (MDAE)
        """

    clf = tree.DecisionTreeRegressor()

    # df = training set, gt = ground truth
    matrix = matrix.dropna(axis=0)
    gt = matrix[ground_truth]
    del matrix[ground_truth]
    df = matrix

    # Split: 90% training, 10% test
    X_train, X_test, y_train, y_test = train_test_split(df, gt, test_size=0.1)

    # initialize scaler
    X_scaler = MinMaxScaler(feature_range=(0.01, 1))
    y_scaler = MinMaxScaler(feature_range=(0.01, 1))

    # shaping
    y_train = np.array(y_train)
    y_train = y_train.reshape(len(y_train), 1)

    # model training
    X_scaler = X_scaler.fit(X_train)
    y_scaler = y_scaler.fit(y_train)

    # scaling
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)
    y_train = y_scaler.transform(y_train)

    # gridsearch for hyperparameter tuning
    from sklearn.model_selection import GridSearchCV

    # metrics for training
    error_metrics = ["neg_mean_squared_error"]

    # range calculated by trail by error
    max_depth_values = [*range(480, 520, 2)]
    max_leaf_values = [*range(700, 800, 2)]

    for error in error_metrics:
        # hyperparams
        parameters = {'max_depth': max_depth_values, 'max_leaf_nodes': max_leaf_values}

        # cross validation
        grid_clf = GridSearchCV(clf, parameters, cv=5, scoring=error)
        grid_clf.fit(X_train, y_train)

        # get optimal values
        results = pd.DataFrame(grid_clf.cv_results_)
        results = results.sort_values(by=['mean_test_score'], ascending=False)
        best_depth = results["param_max_depth"].iloc[0]
        best_leafes = results["param_max_leaf_nodes"].iloc[0]

        # fit optimal value to model
        clf.max_depth = best_depth
        clf.max_leaf_nodes = best_leafes
        clf.fit(X_train, y_train)

        # reverse scaling
        y_pred = clf.predict(X_test)
        y_pred = np.array(y_pred)
        y_pred = y_pred.reshape(len(y_pred), 1)
        y_pred = y_scaler.inverse_transform(y_pred)

        # evaluation by error metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        mdae = median_absolute_error(y_test, y_pred)

    return mse, mae, mape, mdae


def random_forest_regressor(matrix, ground_truth):
    """
    training of random forest regressor and approximation of ground truth
    evaluation by error metrics

    :param matrix: feature matrix
    :param ground_truth: approximated RUMSI / PLT
    :return: error metrics mean squared error (MSE), mean absolute error (MAE), mean absolute percentage error (MAPE),
             median absolute error (MDAE)
    """

    clf = RandomForestRegressor(random_state=0)

    # df = training set, gt = ground truth
    matrix = matrix.dropna(axis=0)
    gt = matrix[ground_truth]
    del matrix[ground_truth]
    df = matrix

    # Split: 90% training, 10% test
    X_train, X_test, y_train, y_test = train_test_split(df, gt, test_size=0.1)

    # initialize scaler
    X_scaler = MinMaxScaler(feature_range=(0.01, 1))
    y_scaler = MinMaxScaler(feature_range=(0.01, 1))

    # shaping
    y_train = np.array(y_train)
    y_train = y_train.reshape(len(y_train), 1)

    # model training
    X_scaler = X_scaler.fit(X_train)
    y_scaler = y_scaler.fit(y_train)

    # scaling
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)
    y_train = y_scaler.transform(y_train)

    # gridsearch for hyperparameter tuning
    from sklearn.model_selection import GridSearchCV

    # metrics for training
    error_metrics = ["neg_mean_squared_error"]

    # range calculated by trail by error
    max_depth_values = [*range(480, 520, 2)]
    max_leaf_values = [*range(700, 800, 2)]

    for error in error_metrics:
        # hyperparams
        parameters = {'max_depth': max_depth_values, 'max_leaf_nodes': max_leaf_values}

        # cross validation
        grid_clf = GridSearchCV(clf, parameters, cv=5, scoring=error)
        grid_clf.fit(X_train, y_train)

        # get optimal values
        results = pd.DataFrame(grid_clf.cv_results_)
        results = results.sort_values(by=['mean_test_score'], ascending=False)
        best_depth = results["param_max_depth"].iloc[0]
        best_leafes = results["param_max_leaf_nodes"].iloc[0]

        # fit optimal value to model
        clf.max_depth = best_depth
        clf.max_leaf_nodes = best_leafes
        clf.fit(X_train, y_train)

        # reverse scaling
        y_pred = clf.predict(X_test)
        y_pred = np.array(y_pred)
        y_pred = y_pred.reshape(len(y_pred), 1)
        y_pred = y_scaler.inverse_transform(y_pred)

        # evaluation by error metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        mdae = median_absolute_error(y_test, y_pred)

    return mse, mae, mape, mdae

