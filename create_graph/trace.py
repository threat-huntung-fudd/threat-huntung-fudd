import json
from logging import Filterer
import logging
from math import e
from re import T
from tqdm import tqdm
from tqdm import tqdm
import csv
import os
import threading
import glob
import pandas as pd
import yaml
import datetime

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# Create a StreamHandler
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

# Add the StreamHandler to the logger
logger.addHandler(stream_handler)

EVENT = "com.bbn.tc.schema.avro.cdm18.Event"
DATUM = "datum"
EVENT_TYPE = "type"
UUID = "com.bbn.tc.schema.avro.cdm18.UUID"
UUID_SHORT = "uuid"

SUBJECT = "com.bbn.tc.schema.avro.cdm18.Subject"
FILEOBJECT = "com.bbn.tc.schema.avro.cdm18.FileObject"
NETFLOW_OBEJCT = "com.bbn.tc.schema.avro.cdm18.NetFlowObject"
# SRCSINK_OBJECT = "com.bbn.tc.schema.avro.cdm18.SrcSinkObject"

SUBOBJ = "subject"
PREDOBJ = "predicateObject"
PREDOBJ2 = "predicateObject2"


class DarpaFileParser:
    def __init__(self):
        self.exclude_nodes = [
            "com.bbn.tc.schema.avro.cdm18.Host",
            "com.bbn.tc.schema.avro.cdm18.TimeMarker",
            "com.bbn.tc.schema.avro.cdm18.StartMarker",
            "com.bbn.tc.schema.avro.cdm18.UnitDependency",
            "com.bbn.tc.schema.avro.cdm18.EndMarker",
            "com.bbn.tc.schema.avro.cdm18.UnnamedPipeObject",
            "com.bbn.tc.schema.avro.cdm18.MemoryObject",
            "com.bbn.tc.schema.avro.cdm18.Principal",
            "com.bbn.tc.schema.avro.cdm18.SrcSinkObject",
        ]
        self.keep_events = [
            "EVENT_EXECUTE",
            "EVENT_SENDTO",
            "EVENT_RECVFROM",
            "EVENT_WRITE",
            "EVENT_READ",
            "EVENT_MODIFY_FILE_ATTRIBUTES",
            # "EVENT_RENAME", # noise event at trace because nearly only firefox does this and this a lot
            "EVENT_RECVMSG",
        ]
        self.extracted_nodes = []
        self.extracted_edges = []
        self.excluded_nodes = set()
        self.seen_edges = set()

    def parse_json_file(self, file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                for line in tqdm(file):
                    self.parse_json_line(line)
        except FileNotFoundError:
            print(f"File '{file_path}' not found.")
        except json.JSONDecodeError:
            print(f"Error parsing JSON file '{file_path}'.")

    def parse_json_line(self, line):
        try:
            transformed_line = json.loads(line)
            if self.check_line_event(transformed_line):
                if self.check_event(transformed_line):
                    event_data = self.extract_event_data(transformed_line)
                    if self.check_excluded_nodes_for_event(event_data):
                        if not self.check_seen_edges(event_data):
                            self.extracted_edges.append(event_data)
                            self.seen_edges.add(
                                (
                                    event_data["subject"],
                                    event_data["type"],
                                    event_data["pred_obj"],
                                    event_data["pred_obj2"],
                                )
                            )
            elif self.check_node(transformed_line):
                extracted_data = self.extract_node_data(transformed_line)
                if extracted_data:
                    self.extracted_nodes.append(extracted_data)
            elif not self.check_node(transformed_line):
                extracted_data = self.extract_node_uuid_excluded(transformed_line)
                if extracted_data:
                    self.excluded_nodes.add(extracted_data)
        except json.JSONDecodeError:
            print(f"Error parsing JSON line '{line}'.")

    def check_seen_edges(self, event_data):
        edge = (
            event_data["subject"],
            event_data["type"],
            event_data["pred_obj"],
            event_data["pred_obj2"],
        )
        return edge in self.seen_edges

    def check_excluded_nodes_for_event(self, event_data):
        if event_data["pred_obj"] in self.excluded_nodes:
            return False
        return True

    def extract_node_uuid_excluded(self, line):
        datum_keys = list(line[DATUM].keys())
        node_type = datum_keys[0]
        if (
            node_type == "com.bbn.tc.schema.avro.cdm18.TimeMarker"
            or node_type == "com.bbn.tc.schema.avro.cdm18.StartMarker"
            or node_type == "com.bbn.tc.schema.avro.cdm18.UnitDependency"
        ):
            return None
        uuid = line[DATUM][node_type][UUID_SHORT]
        return uuid

    def extract_node_data(self, line):
        datum_keys = list(line[DATUM].keys())
        if SUBJECT in datum_keys:
            return self.extract_subject_data(line)
        if FILEOBJECT in datum_keys:
            return self.extract_file_object_data(line)
        if NETFLOW_OBEJCT in datum_keys:
            return self.extract_netflow_object_data(line)
        return None

    def check_line_event(self, line):
        datum_keys = list(line[DATUM].keys())
        if EVENT not in datum_keys:
            return False
        return True

    def check_event(self, event):
        if event[DATUM][EVENT][EVENT_TYPE] in self.keep_events:
            return True
        return False

    def check_node(self, node):
        datum_keys = list(node[DATUM].keys())
        return not any(
            exclude_node in datum_keys for exclude_node in self.exclude_nodes
        )

    def extract_subject_data(self, data):
        subject_uuid = data[DATUM][SUBJECT][UUID_SHORT]
        type = data[DATUM][SUBJECT]["type"]
        if type == "SUBJECT_UNIT":
            return None
        exec = data[DATUM][SUBJECT]["properties"]["map"]["name"]
        return {subject_uuid: {"type": type, "location": exec}}

    def extract_file_object_data(self, data):
        file_uuid = data[DATUM][FILEOBJECT][UUID_SHORT]
        path = data[DATUM][FILEOBJECT]["baseObject"]["properties"]["map"]["path"]
        file_type = data[DATUM][FILEOBJECT]["type"]
        return {file_uuid: {"type": file_type, "location": path}}

    def extract_netflow_object_data(self, data):
        netflow_uuid = data[DATUM][NETFLOW_OBEJCT][UUID_SHORT]
        remote_port = data[DATUM][NETFLOW_OBEJCT]["remotePort"]
        remote_address = data[DATUM][NETFLOW_OBEJCT]["remoteAddress"]
        return {
            netflow_uuid: {
                "type": "NetFlowObject",
                "location": str(remote_address) + ":" + str(remote_port),
            }
        }

    def extract_event_data(self, data):
        event_type = data[DATUM][EVENT][EVENT_TYPE]
        subject = data[DATUM][EVENT][SUBOBJ][UUID]
        pred_obj = data[DATUM][EVENT][PREDOBJ][UUID]
        pred_obj2_tmp = data[DATUM][EVENT][PREDOBJ2]
        if pred_obj2_tmp:
            pred_obj2 = pred_obj2_tmp[UUID]
        else:
            pred_obj2 = None
        timestamp_nanos = data[DATUM][EVENT]["timestampNanos"]
        sequence = data[DATUM][EVENT]["sequence"]["long"]
        return {
            "type": event_type,
            "subject": subject,
            "pred_obj": pred_obj,
            "pred_obj2": pred_obj2,
            "timestamp_nanos": timestamp_nanos,
            "sequence": sequence,
        }

    def write_nodes_to_csv(self, path):
        with open(path, "w", encoding="utf-8", newline="") as nodes_file:
            writer = csv.writer(nodes_file)
            writer.writerow(["UUID", "Type", "Location"])
            for node in self.extracted_nodes:
                uuid = list(node.keys())[0]
                node_data = node[uuid]
                node_type = node_data["type"]
                location = node_data["location"]
                writer.writerow([uuid, node_type, location])

    def write_edges_to_csv(self, path):
        with open(path, "w", encoding="utf-8", newline="") as edges_file:
            writer = csv.writer(edges_file)
            writer.writerow(
                [
                    "Type",
                    "Subject",
                    "PredicateObject",
                    "PredicateObject2",
                    "TimestampNanos",
                    "Sequence",
                ]
            )
            for edge in self.extracted_edges:
                writer.writerow(
                    [
                        edge["type"],
                        edge["subject"],
                        edge["pred_obj"],
                        edge["pred_obj2"],
                        edge["timestamp_nanos"],
                        edge["sequence"],
                    ]
                )

PARSE_SINGLE_FILES = True
MERGE = True
CREATE_DATA = True
CREATE_EVENTLIST = True
GET_GT_INFO = True
CREATE_TGB = True

# Multiple File Parser with multi thread
import concurrent.futures

if __name__ == "__main__":
    logger.debug("Starting parser")
    INPUT_PATH = (
        "./create_graph/input_data/trace/unpacked/"
    )
    OUTPUT_PATH = "./create_graph/output_data/trace/darpa_trace/"
    FILE_NAMES = os.listdir(INPUT_PATH)

    # PARSE_SINGLE_FILES = False
    if PARSE_SINGLE_FILES:

        def process_file(FILE_NAME):
            parser = DarpaFileParser()
            FILE_PATH = os.path.join(INPUT_PATH, FILE_NAME)
            OUTPUT_NODE_NAME = FILE_NAME + "_extracted_nodes.csv"
            OUTPUT_EDGE_NAME = FILE_NAME + "_extracted_edges.csv"
            OUTPUT_NODE_NAME_PATH = os.path.join(OUTPUT_PATH, OUTPUT_NODE_NAME)
            OUTPUT_EDGE_NAME_PATH = os.path.join(OUTPUT_PATH, OUTPUT_EDGE_NAME)
            logger.debug("Reading file '%s'", FILE_PATH)
            parser.parse_json_file(file_path=FILE_PATH)
            parser.write_nodes_to_csv(OUTPUT_NODE_NAME_PATH)
            parser.write_edges_to_csv(OUTPUT_EDGE_NAME_PATH)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(process_file, FILE_NAMES)
    logger.debug("Finished parsing")

    def merge_files(file_type, output_path, header):
        extracted_files = glob.glob(output_path + f"*_extracted_{file_type}s.csv")
        merged_file_path = os.path.join(output_path, f"merged_{file_type}s.csv")
        seen = set()

        with open(merged_file_path, "w", encoding="utf-8", newline="") as merged_file:
            writer = csv.writer(merged_file)
            writer.writerow(header)
            for file in tqdm(extracted_files):
                with open(file, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    next(reader)  # Skip header
                    for row in reader:
                        # Convert the row to a tuple so it can be added to a set
                        row_tuple = tuple(row)
                        if row_tuple not in seen:
                            writer.writerow(row)
                            seen.add(row_tuple)

    # Merge node files
    # MERGE = False
    if MERGE:
        merge_files("node", OUTPUT_PATH, ["UUID", "Type", "Location"])

        # Merge edge files
        merge_files(
            "edge",
            OUTPUT_PATH,
            [
                "Type",
                "Subject",
                "PredicateObject",
                "PredicateObject2",
                "TimestampNanos",
                "Sequence",
            ],
        )

    # Merge and rename merged egde and node file
    def merge_and_rename(
        df_edges_param,
        df_nodes,
        merge_column_left,
        merge_column_right,
        new_column_names,
    ) -> pd.DataFrame:
        df_edges_tmp: pd.DataFrame = pd.merge(
            left=df_edges_param,
            right=df_nodes,
            left_on=[merge_column_left],
            right_on=[merge_column_right],
            how="left",
        )
        df_edges_tmp.rename(columns=new_column_names, inplace=True)
        df_edges_tmp.drop(columns=["UUID"], inplace=True)
        return df_edges_tmp

    # CREATE_DATA = False
    if CREATE_DATA:
        # create eventlist
        merged_nodes_path = os.path.join(OUTPUT_PATH, "merged_nodes.csv")
        df_merged_nodes: pd.DataFrame = pd.read_csv(merged_nodes_path, dtype=str)

        merged_edges_path = os.path.join(OUTPUT_PATH, "merged_edges.csv")
        df_merged_edges = pd.read_csv(merged_edges_path, dtype=str)

        df_merged_edges: pd.DataFrame = merge_and_rename(
            df_edges_param=df_merged_edges,
            df_nodes=df_merged_nodes,
            merge_column_left="Subject",
            merge_column_right="UUID",
            new_column_names={"Type_y": "SubjectType", "Location": "SubjectLocation"},
        )
        df_merged_edges = merge_and_rename(
            df_edges_param=df_merged_edges,
            df_nodes=df_merged_nodes,
            merge_column_left="PredicateObject",
            merge_column_right="UUID",
            new_column_names={
                "Type": "PredicateObjectType",
                "Location": "PredicateObjectLocation",
            },
        )
        df_merged_edges = merge_and_rename(
            df_edges_param=df_merged_edges,
            df_nodes=df_merged_nodes,
            merge_column_left="PredicateObject2",
            merge_column_right="UUID",
            new_column_names={
                "Type": "PredicateObject2Type",
                "Location": "PredicateObject2Location",
            },
        )

        # create temporal edges from eventlist
        columns = (
            [
                "Subject",
                "PredicateObject",
                "Type",
                "TimestampNanos",
                "Sequence",
                "SubjectType",
                "SubjectLocation",
                "DestinationType",
                "DestinationLocation",
            ],
        )
        # Create two dataframes based on the PredicateObject and PredicateObject2 conditions
        df1: pd.DataFrame = df_merged_edges[
            df_merged_edges["PredicateObjectType"].notna()
        ].copy()
        df2: pd.DataFrame = df_merged_edges[
            df_merged_edges["PredicateObject2Type"].notna()
        ].copy()

        df1.drop(
            columns=[
                "PredicateObject2",
                "PredicateObject2Type",
                "PredicateObject2Location",
            ],
            inplace=True,
        )
        df2.drop(
            columns=[
                "PredicateObject",
                "PredicateObjectType",
                "PredicateObjectLocation",
            ],
            inplace=True,
        )
        # Rename columns for df1
        df1 = df1.rename(
            columns={
                "PredicateObject": "PredicateObject",
                "Type_x": "Type",
                "PredicateObjectType": "DestinationType",
                "PredicateObjectLocation": "DestinationLocation",
            }
        )

        # Rename columns for df2
        df2 = df2.rename(
            columns={
                "PredicateObject2": "PredicateObject",
                "Type_x": "Type",
                "PredicateObject2Type": "DestinationType",
                "PredicateObject2Location": "DestinationLocation",
            }
        )

        # Concatenate the two dataframes
        data: pd.DataFrame = pd.concat([df1, df2])

        # remove rows, where SubjectLocation is nan
        data = data.dropna(subset=["SubjectLocation"])

        data.to_csv(os.path.join(OUTPUT_PATH, "trace_data.csv"), index=False)
        logger.debug("Length of data: %s", len(data))
        logger.debug("Created trace_data.csv")

    if CREATE_EVENTLIST:
        data_path = os.path.join(OUTPUT_PATH, "trace_data.csv")

        df_edges_output = pd.read_csv(data_path, dtype=str)
        # df_edges_output = pd.DataFrame(data=data, dtype=str)
        logger.debug("Length of df_edges_output: %s", len(df_edges_output))
        df_edges_output.drop_duplicates(
            subset=[
                "Type",
                "SubjectType",
                "SubjectLocation",
                "DestinationType",
                "DestinationLocation",
            ],
            inplace=True,
            keep="first",
        )
        logger.debug(
            "Length of df_edges_output after deduplication: %s", len(df_edges_output)
        )
        # Sort df_edges_output by sequence
        df_edges_output["Sequence"] = df_edges_output["Sequence"].astype(int)
        # df_edges_output.sort_values(by="Sequence", inplace=True, ascending=True)
        df_edges_output["TimestampNanos"] = df_edges_output["TimestampNanos"].astype(
            int
        )
        # df_edges_output.sort_values(
        #     by=["Sequence", "TimestampNanos"], inplace=True, ascending=True
        # )
        df_edges_output.sort_values(by="TimestampNanos", inplace=True, ascending=True)
        logger.debug("Length of df_edges_output: %s", len(df_edges_output))
        filter_noise_file_types = True
        if filter_noise_file_types:
            # drop rows where destination type is noise (FILE_OBJECT_CHAR, FILE_OBJECT_BLOCK, FILE_OBJECT_DIR)
            noise_file_types = [
                "FILE_OBJECT_CHAR",
                "FILE_OBJECT_BLOCK",
                "FILE_OBJECT_DIR",
            ]
            df_edges_output = df_edges_output[
                ~df_edges_output["DestinationType"].isin(noise_file_types)
            ]
        filter_noise = False
        if filter_noise:
            filter_strings = [
                ".so.",
                "/proc",
                "/dev",
                ".bash_history",
                "bash_completion.d",
                "/run/motd.dynamic",
                "/bin/sh",
                ".bash_logout",
                "firefox",  # firefox is part of the attack and cant be seen as noise
            ]

            for string in filter_strings:
                df_edges_output = df_edges_output[
                    ~df_edges_output["DestinationLocation"].str.contains(
                        string, regex=True
                    )
                ]
                df_edges_output = df_edges_output[
                    ~df_edges_output["SubjectLocation"].str.contains(string, regex=True)
                ]
                logger.debug("After filtering %s: %s", string, len(df_edges_output))

        logger.debug("Length of df_edges_output: %s", len(df_edges_output))

        logger.debug("Length of df_edges_output: %s", len(df_edges_output))
        logger.debug(
            "Unique SubjectLocations: %s",
            len(df_edges_output["SubjectLocation"].unique()),
        )
        logger.debug(
            "Unique DestinationLocations: %s",
            len(df_edges_output["DestinationLocation"].unique()),
        )
        # save events
        df_edges_output.to_csv(os.path.join(OUTPUT_PATH, "events.csv"), index=False)

    def parse_date(date, time) -> int:
        ts = datetime.datetime.strptime(date + " " + time, "%Y%m%d %H:%M")
        ts = ts + datetime.timedelta(hours=4) # depents on system time zone
        timestamp_ns = ts.timestamp() * 1_000_000_000
        return int(timestamp_ns)

    # get gt info

    if GET_GT_INFO:
        gt_metas = [
            "./create_graph/configs/trace/darpa-trace-00_gt_meta.yaml",
            "./create_graph/configs/trace/darpa-trace-01_gt_meta.yaml",
            "./create_graph/configs/trace/darpa-trace-02_gt_meta.yaml",
            "./create_graph/configs/trace/darpa-trace-03_gt_meta.yaml",
        ]
        # Get GT information
        gt_start_times = []
        gt_end_times = []
        gt_search_terms = []
        for attack_file in gt_metas:
            with open(attack_file, "r") as file:
                yaml_data = yaml.safe_load(file)
            date = yaml_data["date"]
            start_time = yaml_data["start_time"]
            end_time = yaml_data["end_time"]
            gt_start_times.append(parse_date(date, start_time))
            gt_end_times.append(parse_date(date, end_time))
            gt_search_terms.append(yaml_data["search_terms"])

    # bring it in TGB format
    if CREATE_TGB:
        # Load the events
        events = pd.read_csv(os.path.join(OUTPUT_PATH, "events.csv"), dtype=str)

        events.sort_values(by="TimestampNanos", inplace=True)

        events.drop(columns=["Subject", "PredicateObject"], inplace=True)

        new_column_sequence = [
            "SubjectLocation",
            "DestinationLocation",
            "Sequence",
            "Label",
            "SubjectType",
            "DestinationType",
            "Type",
            "TimestampNanos",
        ]
        events = events.reindex(columns=new_column_sequence)

        events["TimestampNanos"] = events["TimestampNanos"].astype(int)
        events.sort_values(by="TimestampNanos", inplace=True, ascending=True)

        events["Sequence"] = (
            events.reset_index().index
        )  # reset index to get a monotonic increasing sequence

        def label_row(row):
            # Add the label column
            for i in range(len(gt_start_times)):
                if gt_start_times[i] <= int(row.TimestampNanos) <= gt_end_times[i]:
                    for term in gt_search_terms[i]:
                        if (
                            term in row.SubjectLocation
                            or term in row.DestinationLocation
                        ):
                            return "0"
            return "1"

        logger.debug("Start Labeling")

        events["Label"] = events.apply(lambda row: label_row(row), axis=1)
        logger.debug(" Label 0: %s", len(events[events["Label"] == "0"]))
        logger.debug(" Label 1: %s", len(events[events["Label"] == "1"]))
        logger.debug("Finished Labeling")

        events.to_csv(
            os.path.join(OUTPUT_PATH, "events_tgb_normal.csv"),
            index=False,
            header=False,
        )

        def transform_column(df, column_name):
            categories = df[column_name].unique()
            categories_dict = {categories[i]: i for i in range(len(categories))}
            df[column_name] = df[column_name].map(categories_dict)
            return df

        events.drop(columns=["TimestampNanos"], inplace=True)
        events = transform_column(events, "Type")
        events = transform_column(events, "SubjectType")
        events = transform_column(events, "DestinationType")
        events = transform_column(events, "SubjectLocation")
        events = transform_column(events, "DestinationLocation")

        offset = len(events["SubjectLocation"].unique())
        events["DestinationLocation"] = events["DestinationLocation"] + offset


        events.to_csv(
            os.path.join(OUTPUT_PATH, "events_tgb_mapped.csv"),
            index=False,
            header=False,
        )
    logger.debug("Finished adding new columns to merged_edges")

    logger.debug("Finished merging")
