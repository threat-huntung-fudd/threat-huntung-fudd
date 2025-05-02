import copy
import logging
import timeit
import time
import datetime
import os
import math
from flask.cli import F
from tqdm import tqdm
import numpy as np
import warnings
import shutil
import json
import torch
import torch.nn as nn
import mlflow
import itertools
from types import SimpleNamespace
from omegaconf import OmegaConf, ListConfig

from models.TGAT import TGAT
from models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from models.CAWN import CAWN
from models.TCL import TCL
from models.GraphMixer import GraphMixer
from models.DyGFormer import DyGFormer
from models.modules import ContextualLinkPredictor
from utils.utils import (
    set_random_seed,
    convert_to_gpu,
    get_parameter_sizes,
    create_optimizer,
)
from utils.utils import get_neighbor_sampler
from utils.metrics import get_link_prediction_metrics
from utils.DataLoader import (
    get_idx_data_loader,
    get_link_anom_det_data_TRANS_TGB,
)
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import (
    get_config_for_anomaly_detection,
    update_config_anomaly_detection,
)

import tgb
from tgb.linkanomdet.evaluate import Evaluator
from evaluation.tgb_evaluate_linkanomdet import eval_linkanomdet_TGB
from sklearn.metrics import classification_report, f1_score
from sklearn.metrics import confusion_matrix
import pandas as pd

import matplotlib.pyplot as plt


def reverse_parse_date(timestamp_ns: int) -> (datetime, str, str):
    ts = datetime.datetime.fromtimestamp(
        timestamp_ns / 1_000_000_000
    ) - datetime.timedelta(hours=6)
    date_str = ts.strftime("%Y%m%d")
    time_str = ts.strftime("%H:%M")
    return ts, date_str, time_str


def parse_date(date, time) -> int:
    ts = datetime.datetime.strptime(date + " " + time, "%Y%m%d %H:%M")
    ts = ts + datetime.timedelta(hours=6)
    timestamp_ns = ts.timestamp() * 1_000_000_000
    return int(timestamp_ns)


def result_analysis(args: SimpleNamespace):
    # Silence PyTorch warnings.
    warnings.filterwarnings("ignore")
    # Silence a warning because there is no git executable.
    os.environ["GIT_PYTHON_REFRESH"] = "quiet"
    args.experiment_name = f"{args.experiment_name}/{args.val_anom_type}-{args.test_anom_type}/{args.dataset_name}/{args.model_name}"

    # read y_pred and y_label
    read_result_folder = f"{args.output_root}/saved_results/{args.experiment_name}"
    args.save_model_name = f"{args.model_name}_{args.dataset_name}_lr_{args.learning_rate}_seed_{args.seed}_set_{args.anom_set_id}_run_{args.run_id}"
    # /home/buchta/sambashare/gnn_darpa_lab/TGB-link-anomaly-detection-baselines/experiments/saved_results/Example Cadets TT/organic-organic/darpa-cadets-tt-01/DyGFormer/DyGFormer_darpa-cadets-tt-01_lr_0.0001_seed_2024_set_0_run_0_y_pred.txt
    read_result_pred_path = os.path.join(
        read_result_folder, f"{args.save_model_name}_y_pred.txt"
    )

    read_result_label_path = os.path.join(
        read_result_folder, f"{args.save_model_name}_y_label.txt"
    )

    y_pred = np.loadtxt(read_result_pred_path)
    y_label = np.loadtxt(read_result_label_path)
    unique_value_count = np.unique(y_label, return_counts=True)
    print("Unique value count of y_label:")
    for value, count in zip(unique_value_count[0], unique_value_count[1]):
        print(f"{value}: {count}")

    print(f"len of y_pred: {len(y_pred)}")
    print(f"len of y_label: {len(y_label)}")

    # compare y_pred and y_label
    # Convert y_pred to binary

    y_pred_binary = np.where(y_pred >= args.THRESHOLD, 1, 0)

    cl_report = classification_report(y_label, y_pred_binary)
    print(cl_report)

    confusion_mat = confusion_matrix(y_label, y_pred_binary)

    tn, fp, fn, tp = confusion_mat.ravel()
    print(f"tn: {tn}, fp: {fp}, fn: {fn}, tp: {tp}")

    f1_s = f1_score(y_label, y_pred_binary)
    print(f"F1 Score: {f1_s}")

    # get the top k anomalous events
    k = 5000
    top_k_anomalous_events = np.argsort(y_pred)[:k]
    # check if the top k anomalous events are actually anomalous
    top_k_anomalous_events_labels = y_label[top_k_anomalous_events]
    print(f"Top {k} Anomalous Events Labels: {top_k_anomalous_events_labels}")

    # Calculate the percentage of top k events that are anomalies
    percent_anomalies = ((k - np.sum(top_k_anomalous_events_labels)) / k) * 100
    print(f"Percentage of top {k} events that are anomalies: {percent_anomalies}%")

    path_raw_eventlist = (
        f"{args.output_root}/../data/{args.experiment_name}/events_tgb_normal.csv"
    )
    dsn = args.dataset_name
    dsn_ = dsn.replace("-", "_")

    # path_raw_eventlist = f"{args.output_root}/../data/{dsn_}/cadets_normal.csv"
    path_raw_eventlist = f"{args.dataset_root}/{dsn_}/events_tgb_normal.csv"

    print(path_raw_eventlist)

    # Read eventlist into pandas DataFrame
    col_names = [
        "exec",
        "location",
        "sequence",
        "edge_label",
        "source_type",
        "destination_type",
        "event_type",
        "ts",
    ]
    df = pd.read_csv(
        filepath_or_buffer=path_raw_eventlist, header=None, names=col_names
    )

    # cut train and val from df to get the same length as y_pred
    df = df.tail(len(y_pred))
    df["orig_index_column"] = df.index
    df = df.reset_index(drop=True)
    # get execs of the top k anomalous events

    import datetime

    def reverse_parse_date(timestamp_ns: int) -> str:
        ts = datetime.datetime.fromtimestamp(
            timestamp_ns / 1_000_000_000
        ) - datetime.timedelta(hours=6)
        ts = ts.strftime("%Y%m%d %H:%M")
        return ts

    df["ts_h"] = df["ts"].apply(func=lambda x: reverse_parse_date(timestamp_ns=x))

    top_k_anomalous_execs = np.unique(df.iloc[top_k_anomalous_events]["exec"].values)
    print(f"Top {k} Anomalous Execs: {top_k_anomalous_execs}")

    # df["sequence"] = df["sequence"].astype(int)
    df["ts"] = df["ts"].astype(int)
    print(f"lowest timestamp: {int(df['ts'].min())}")
    print(f"highest timestamp: {int(df['ts'].max())}")
    # print(f"highest timestamp: {int(df['sequence'].max())}")

    # get df_rows of the top k anomalous events
    top_k_anomalous_df_rows: pd.DataFrame = df.iloc[top_k_anomalous_events]
    top_k_anomalous_df_rows_: pd.DataFrame = df.iloc[top_k_anomalous_events]

    print(f"Top {k} Anomalous DF Rows: {top_k_anomalous_df_rows}")

    # filter noise files like starting with /home/admin/.cache/mozilla/firefox/
    # top_k_anomalous_df_rows = top_k_anomalous_df_rows[
    #     ~top_k_anomalous_df_rows["location"].str.startswith(
    #         "/home/admin/.cache/mozilla/firefox/"
    #     )
    # ]  # from 5000 to 672
    # filter noise exec = firefox or cache
    top_k_anomalous_df_rows = top_k_anomalous_df_rows[
        ~top_k_anomalous_df_rows["exec"].str.contains("firefox|Cache2 I/O")
    ]  # from 5000 to 672

    # top_k_anomalous_df_rows_with_sus_path = top_k_anomalous_df_rows_[
    #     top_k_anomalous_df_rows_["location"].str.startswith(
    #         "/home/admin/.cache/mozilla/firefox/"
    #     )
    # ]  # from 5000 to 672
    top_k_anomalous_df_rows_with_sus_path = top_k_anomalous_df_rows_[
        top_k_anomalous_df_rows_["exec"].str.contains("firefox|Cache2 I/O")
    ]  # from 5000 to 672

    # group by exec
    top_k_anomalous_df_rows_grouped = (
        top_k_anomalous_df_rows.groupby("exec").size().reset_index(name="count")
    )

    # group by 15min time windows
    top_k_anomalous_df_rows["ts_h"] = pd.to_datetime(
        arg=top_k_anomalous_df_rows["ts_h"]
    )
    top_k_anomalous_df_rows["ts_h_60"] = top_k_anomalous_df_rows["ts_h"].dt.floor(
        freq="60min"
    )
    top_k_anomalous_df_rows_["ts_h"] = pd.to_datetime(
        arg=top_k_anomalous_df_rows_["ts_h"]
    )

    top_k_anomalous_df_rows_["ts_h_60"] = top_k_anomalous_df_rows_["ts_h"].dt.floor(
        freq="60min"
    )
    top_five = top_k_anomalous_df_rows_["ts_h_60"].value_counts().head(5)
    print(f"Top 5 Timeframes: {top_five}")

    top_k_anomalous_df_rows_grouped_time = (
        top_k_anomalous_df_rows.groupby("ts_h_60").size().reset_index(name="count")
    )

    # group by 15min time windows
    top_k_anomalous_df_rows_with_sus_path["ts_h"] = pd.to_datetime(
        top_k_anomalous_df_rows_with_sus_path["ts_h"]
    )
    top_k_anomalous_df_rows_with_sus_path["ts_h_60"] = (
        top_k_anomalous_df_rows_with_sus_path["ts_h"].dt.floor(freq="60min")
    )
    top_k_anomalous_df_rows_with_sus_path_groubed_time = (
        top_k_anomalous_df_rows_with_sus_path.groupby("ts_h_60")
        .size()
        .reset_index(name="count")
    )

    # show unique execs
    print(f"Unique Execs: {top_k_anomalous_df_rows['exec'].unique()}")
    print(f"Unique Execs Count: {len(top_k_anomalous_df_rows['exec'].unique())}")
    print(f"Value Count of Execs: {top_k_anomalous_df_rows['exec'].value_counts()}")

    # show unique files
    print(f"Unique Files: {top_k_anomalous_df_rows['location'].unique()}")
    print(f"Unique Files Count: {len(top_k_anomalous_df_rows['location'].unique())}")

    # show count of unique files
    print(f"Unique Files Count: {top_k_anomalous_df_rows['location'].value_counts()}")

    # print files occuring more than once
    print(
        f"Files occuring more than once: {top_k_anomalous_df_rows['location'].value_counts()[top_k_anomalous_df_rows['location'].value_counts() > 1]}"
    )
    print()

    # save the top k anomalous events to a csv file
    top_k_df_save_path = os.path.join(
        read_result_folder, f"{args.save_model_name}_top_k_anomalous_events.csv"
    )
    print(f"Saving top k anomalous events to: {top_k_df_save_path}")
    top_k_anomalous_df_rows.to_csv(top_k_df_save_path, index=True)
    # top_k_indices = top_k_anomalous_df_rows.index.values.tolist()
    top_k_indices = top_k_anomalous_df_rows["orig_index_column"].values.tolist()
    top_k_indices_str = [str(index) for index in top_k_indices]
    print(f"Top {k} Anomalous Indices: {top_k_indices_str}")

    def calculate_confusion_matrix(y_pred, y_label, min_x):
        min_x_y_pred = np.partition(y_pred, min_x)[: min_x + 1]
        import sys

        epsilon = sys.float_info.epsilon

        threshold = min_x_y_pred[min_x]  # + epsilon
        # print(f"Threshold: {threshold}")
        # print(f"min_x_y_pred: {min_x_y_pred[min_x]}")
        y_pred_binary = np.where(
            y_pred < threshold, 0, 1
        )  # y_pred <= dont cover special case if th = 0
        confusion_mat = confusion_matrix(y_label, y_pred_binary)
        tn, fp, fn, tp = confusion_mat.ravel()
        if tn > 0 or fn > 0:

            # output += f"Confusion Matrix: {confusion_mat}\n"
            output = f"tn: {tn}, fp: {fp}, fn: {fn}, tp: {tp}"
            output += f" Threshold: {threshold}, Min_x: {min_x}\n"
            # print(f"Threshold: {threshold}, Min_x: {min_x}")
            # print(f"Confusion Matrix: {confusion_mat}")
            print(output)

    # Call the function with the desired value for min_x
    calculate_confusion_matrix(y_pred, y_label, 10)
    calculate_confusion_matrix(y_pred, y_label, 25)
    calculate_confusion_matrix(y_pred, y_label, 50)
    calculate_confusion_matrix(y_pred, y_label, 100)
    calculate_confusion_matrix(y_pred, y_label, 500)
    calculate_confusion_matrix(y_pred, y_label, 1000)
    calculate_confusion_matrix(y_pred, y_label, 5000)
    calculate_confusion_matrix(y_pred, y_label, 10000)
    # calculate_confusion_matrix(y_pred, y_label, 20000)
    # calculate_confusion_matrix(y_pred, y_label, 40000)


def main():
    # Get the path to configuration file of the experiment.
    config_path = get_config_for_anomaly_detection("TRACE").config_path
    config = OmegaConf.load(config_path)

    # Add missing common hyperparameters to the config.
    common_args: dict = update_config_anomaly_detection(config)

    # Run on a GPU if available.
    common_args["device"] = f"cuda" if torch.cuda.is_available() else "cpu"
    common_args["mlflow_tracking_uri"] = os.path.abspath(common_args["mlflow_tracking_uri"])
    common_args["output_root"] = os.path.abspath(common_args["output_root"])
    common_args["dataset_root"] = os.path.abspath(common_args["dataset_root"])
    

    # Set up MLflow.
    mlflow.set_tracking_uri(f'file://{common_args["mlflow_tracking_uri"]}')

    # If config does not contain ´val_anom_type´ and ´test_anom_type´, set them to ´anom_type´.
    if config.data.get("val_anom_type") is None:
        if config.data.get("anom_type") is None:
            raise ValueError(
                "val_anom_type and anom_type are both unset! Please specify at least one of them."
            )
        else:
            config.data.val_anom_type = config.data.anom_type

    if config.data.get("test_anom_type") is None:
        if config.data.get("anom_type") is None:
            raise ValueError(
                "test_anom_type and anom_type are both unset! Please specify at least one of them."
            )
        else:
            config.data.test_anom_type = config.data.anom_type

    # Convert common grid search hyperparameters to lists.
    if not isinstance(config.data.dataset_name, ListConfig):
        config.data.dataset_name = ListConfig([config.data.dataset_name])

    if not isinstance(config.data.val_anom_type, ListConfig):
        config.data.val_anom_type = ListConfig([config.data.val_anom_type])

    if not isinstance(config.data.test_anom_type, ListConfig):
        config.data.test_anom_type = ListConfig([config.data.test_anom_type])

    if not isinstance(config.training.learning_rate, ListConfig):
        config.training.learning_rate = ListConfig([config.training.learning_rate])

    for (
        dataset_name,
        (val_anom_type, test_anom_type),
        learning_rate,
    ) in itertools.product(
        config.data.dataset_name,
        zip(config.data.val_anom_type, config.data.test_anom_type),
        config.training.learning_rate,
    ):
        for model_name in config.models.keys():
            # Create MLflow experiment so that all concurent runs can read the experiment ID.
            experiment_name = f'{common_args["experiment_name"]}/{val_anom_type}-{test_anom_type}/{dataset_name}/{model_name}'
            experiment_id = mlflow.get_experiment_by_name(experiment_name)
            if experiment_id is None:
                experiment_id = mlflow.create_experiment(experiment_name)

            # Convert model-specific hyperparameters to lists for grid search.
            for param in config.models[model_name].keys():
                if not isinstance(config.models[model_name][param], ListConfig):
                    config.models[model_name][param] = ListConfig(
                        [config.models[model_name][param]]
                    )

            # Cartesian product of all values of all hyperparameters
            all_model_configs = [
                dict(zip(config.models[model_name].keys(), x))
                for x in itertools.product(*config.models[model_name].values())
            ]

            for model_config in all_model_configs:
                for anom_set_id in range(common_args["num_anom_sets"]):
                    for run_id in range(common_args["num_runs"]):
                        args = copy.copy(common_args)
                        # Update run-specific hyperparameters.
                        args["dataset_name"] = dataset_name
                        args["val_anom_type"] = val_anom_type
                        args["test_anom_type"] = test_anom_type
                        args["learning_rate"] = learning_rate
                        args["model_name"] = model_name
                        args["model_name"] = model_name
                        args["anom_set_id"] = anom_set_id
                        args["run_id"] = run_id

                        args["delete_null_rows"] = False
                        args["reduced_graph"] = False
                        args["delete_logging"] = False
                        args["THRESHOLD"] = (
                            0.5  # 0.0000005  # 0.0000005  # 0.0000005#0.5 #0.1 #0.5 #0.0000005 # Threshold for binary classification of anomaly detection
                        )

                        args["exec_path"] = True

                        args["dataset_root"] = config.data.get("dataset_root")

                        args["only_testing"] = (
                            False  # dont work properly, because the last model is overwritten while staring, and therefore cant be loaded anymore
                        )
                        args.update(model_config)

                        # Convert args from dictonary to SimpleNamespace.
                        print(f"####MODEL: {model_name}####")
                        args = SimpleNamespace(**args)

                        # Run experiment.
                        result_analysis(args=args)


if __name__ == "__main__":
    main()
