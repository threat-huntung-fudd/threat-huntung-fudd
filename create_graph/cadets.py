"""
Parser for DARPA datasets
"""

import logging
import configparser
import re
import os
import datetime
from unittest import skip
from flask import config
from matplotlib import lines
import yaml
import pandas as pd


# Set up logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
# Get the logger
logger = logging.getLogger(__name__)


class DarpaParser:
    """
    Parser for DARPA datasets
    """

    def __init__(self, dataset="cadets", config_type="all", uuid_writer=False):
        """
        Initialize the parser
        """
        self.EVENT = "event"
        self.MEMORYOBJECT = "memoryobject"
        self.NETFLOWOBJECT = "netflowobject"
        self.UNNAMEDPIPEOBJECT = "unnamedpipeobject"
        self.UUID = "uuid"
        self.TYPE = "type"
        self.TIME = "time"
        self.EXEC = "exec"
        self.SRC = "src"
        self.DST1 = "dst1"
        self.DST2 = "dst2"
        self.PATH = "path"
        self.SOCKET = "socket"
        self.IP = "ip"
        self.PORT = "port"

        self.dataset = dataset  # Dataset to parse
        self.config = configparser.ConfigParser()
        self.nodes = {}  # Nodes dictionary
        self.uuid_writer = uuid_writer  # Write uuids instead of indexpos
        self.config_type = config_type

        self.no_not_events = 0
        self.no_not_src = 0
        self.no_not_src_in_nodes = 0
        self.no_not_dst = 0
        self.no_not_dst_in_nodes = 0
        self.no_events = 0

        self.line_count_in_nodes = 0
        self.node_count = 0
        self.line_count_in_edges = 0
        self.event_count = 0
        self.correct_event_count = 0

        # Access the values of the arguments
        logger.info("Dataset: %s", self.dataset)
        # Read the configuration file based on the chosen dataset
        if self.config_type == "00":
            self.config_filename = "./configs/cadets/cadets_e3_exec_00.ini"
        elif self.config_type == "01":
            self.config_filename = "./configs/cadets/cadets_e3_exec_01.ini"
        elif self.config_type == "02":
            self.config_filename = "./configs/cadets/cadets_e3_exec_02.ini"
        elif self.config_type == "all":
            self.config_filename = "./configs/cadets/cadets_e3_exec_all.ini"
        else:
            logger.error(f"Invalid config type: {self.config_type}")
            return
        logger.info(f"Read configuration file: {self.config_filename}")
        self.config.read(self.config_filename)

        # Read the metadata
        (
            self.name,
            self.path,
            self.track,
            self.outputpath,
            self.skip_execs,
            self.keep_events,
            self.gt_metas,
        ) = self.get_metadata()
        logger.info(
            f"Name: {self.name}, Path: {self.path}, Track: {self.track}, Outputpath: {self.outputpath}"
        )
        logger.info(
            f"Skip Events: {self.skip_execs}, Keep Events: {self.keep_events}, GT Metas: {self.gt_metas}"
        )

        # Get the regex expressions
        self.regex_expressions = self.get_regex_expressions("regex_expressions")
        self.exclude_patterns = self.get_regex_expressions("exclude_patterns")
        self.row_types = self.get_regex_expressions("row_types")

        # Get the directory path
        self.file_path = self.path + self.track

        self.files = self.config_array_to_array(self.config.get("track_files", "files"))

        # Get GT information
        self.gt_start_times = []
        self.gt_end_times = []
        self.gt_search_terms = []
        for attack_file in self.gt_metas:
            with open(attack_file, "r") as file:
                self.yaml_data = yaml.safe_load(file)
            date = self.yaml_data["date"]
            start_time = self.yaml_data["start_time"]
            end_time = self.yaml_data["end_time"]
            self.gt_start_times.append(self.parse_date(date, start_time))
            self.gt_end_times.append(self.parse_date(date, end_time))
            self.gt_search_terms.append(self.yaml_data["search_terms"])

        self.unique_edges = set()

        logger.info("Ready for parsing")

    def parse_date(self, date, time) -> int:
        ts = datetime.datetime.strptime(date + " " + time, "%Y%m%d %H:%M")
        ts = ts + datetime.timedelta(hours=4) # sometimes 4 / 6 depents on the timezone # should be 441
        timestamp_ns = ts.timestamp() * 1_000_000_000
        return int(timestamp_ns)

    def parse_data_threatrace(self):
        logger.info("Parsing data based on configuration")
        self.output_eventlist = open(self.outputpath + "eventlist.txt", "w")

        for i in self.files:
            now_path = self.path + i
            file = open(now_path, "r")
            for line in file:
                self.line_count_in_nodes += 1
                self.process_nodes_threatrace(line)
            file.close()
        logger.info(f"Nodes processed: {len(self.nodes)}")
        logger.info(
            f"Number of read lines in nodes preprocessing: {self.line_count_in_nodes}"
        )
        logger.info(f"Number of nodes: {self.node_count}")
        logger.info(
            f"Therefore {self.node_count - len(self.nodes)} nodes were excluded"
        )

        for i in self.files:
            now_path = self.path + i
            if not os.path.exists(now_path):
                break
            file = open(now_path, "r")
            for line in file:
                self.line_count_in_edges += 1
                self.process_edges_threatrace(line)
            file.close()
            logger.info(f"Edges processed in File: {now_path}")

        logger.info(f"Number of events: {self.event_count}")
        logger.info(f"Processed lines for edges: {self.line_count_in_edges}")
        logger.info(f"Number of correct events: {self.correct_event_count}")
        logger.info(
            f"therefore {self.event_count - self.correct_event_count} events were excluded"
        )
        logger.info(f"Number of unique edges: {len(self.unique_edges)}")
        logger.info(
            f"Therefore {self.correct_event_count - len(self.unique_edges)} edges were excluded from the correct events"
        )
        logger.info(f"All edges processed")
        self.output_eventlist.close()
        logger.info(f"Output eventlist file closed")

    def process_nodes_threatrace(self, line):
        if self.row_types[self.EVENT].search(line):
            return
        self.node_count += 1
        for key in self.exclude_patterns:
            if self.exclude_patterns[key].search(line):
                return
        if len(self.regex_expressions[self.UUID].findall(line)) > 0:
            n_uuid = self.regex_expressions[self.UUID].findall(line)[0]
        n_subject_type = self.regex_expressions[self.TYPE].findall(line)
        if len(n_subject_type) < 1:

            if self.row_types[self.NETFLOWOBJECT].search(line):
                self.nodes[n_uuid] = {
                    "type": self.NETFLOWOBJECT,
                    "indexpos": len(self.nodes),
                }
                return

        self.nodes[n_uuid] = {"type": n_subject_type[0], "indexpos": len(self.nodes)}

    def process_edges_threatrace(self, line):
        if not self.row_types[self.EVENT].search(line):
            self.no_not_events += 1
            return
        self.event_count += 1
        e_type = self.regex_expressions[self.TYPE].findall(line)[0]
        e_time = self.regex_expressions[self.TIME].findall(line)[0]
        e_exec = self.regex_expressions[self.EXEC].findall(line)
        e_src = self.regex_expressions[self.SRC].findall(line)
        e_ip = self.regex_expressions[self.IP].findall(line)
        e_port = self.regex_expressions[self.PORT].findall(line)
        e_path = self.regex_expressions[self.PATH].findall(line)

        fields = [e_type, e_time, e_exec, e_src, e_ip, e_port, e_path]

        if len(e_src) < 1:
            self.no_not_src += 1
            return
        e_src = e_src[0]
        if e_src not in self.nodes:
            self.no_not_src_in_nodes += 1
            return
        if len(e_exec) < 1:
            e_exec = "null"
        else:
            e_exec = e_exec[0]
        if len(e_ip) < 1:
            e_ip = "null"
        else:
            e_ip = e_ip[0].replace(",", "")
        if len(e_port) < 1:
            e_port = "null"
        else:
            e_port = e_port[0].replace(",", "")
        if len(e_path) < 1:
            e_path = "null"
        else:
            e_path = e_path[0].replace(",", "")
        src_type = self.nodes[e_src]["type"]
        src_indexpos = self.nodes[e_src]["indexpos"]
        e_dst1 = self.regex_expressions[self.DST1].findall(line)
        e_dst2 = self.regex_expressions[self.DST2].findall(line)

        if e_exec == "null":
            return
        if e_type not in self.keep_events:
            return
        if e_exec in self.skip_execs:
            return

        for e_dst in [e_dst1, e_dst2]:
            if len(e_dst) > 0 and e_dst[0] != "null":
                e_dst = e_dst[0]
                if e_dst not in self.nodes:
                    self.no_not_dst_in_nodes += 1
                    return
                dst_type = self.nodes[e_dst]["type"]
                dst_indexpos = self.nodes[e_dst]["indexpos"]

                self.correct_event_count += 1
                # filter duplicates (keep first)
                if (
                    src_type,
                    e_exec,
                    e_dst,
                    dst_type,
                    e_type,
                    e_path,
                    e_ip,
                    e_port,
                ) in self.unique_edges:
                    return
                self.unique_edges.add(
                    (src_type, e_exec, e_dst, dst_type, e_type, e_path, e_ip, e_port)
                )

                if self.uuid_writer:
                    edge = f"{e_src},{src_type},{e_exec},{e_dst},{dst_type},{e_type},{e_time}, {e_path}, {e_ip}, {e_port}\n"
                else:
                    edge = f"{src_indexpos},{src_type},{e_exec},{dst_indexpos},{dst_type},{e_type},{e_time}, {e_path}, {e_ip}, {e_port}\n"
                self.output_eventlist.write(edge)

            else:
                self.no_not_dst += 1

    def get_metadata(self) -> tuple:
        logger.info("Reading metadata")
        name = self.config.get("dataset", "name")
        path = self.config.get("dataset", "path")
        track = self.config.get("dataset", "track")
        outputpath = self.config.get("dataset", "outputpath")
        skip_execs = self.config_array_to_array(
            self.config.get("skip_execs", "skip_execs")
        )
        keep_events = self.config_array_to_array(
            self.config.get("keep_events", "keep_events")
        )
        gt_files = self.config_array_to_array(self.config.get("files", "gt_metas"))
        return name, path, track, outputpath, skip_execs, keep_events, gt_files

    def config_array_to_array(self, configline) -> list:
        configline = configline.replace("[", "")
        configline = configline.replace("]", "")
        configline = configline.replace("'", "")
        configline = configline.replace('"', "")
        configline = configline.replace(" ", "")
        return configline.split(",")

    def get_regex_expressions(self, tag) -> dict:
        logger.info("Getting regex expressions")
        regex_section = self.config[tag]
        regex_expressions = {}
        for ex in regex_section:
            key = ex
            value = r"{}".format(regex_section[ex])
            value = value[2:-1]  # remove r'/ and ' from the string
            try:
                regex = re.compile(value)
                regex_expressions[key] = regex
                logger.info(f"Regex expression added: {key} - {value}")
            except re.error as e:
                logger.error(f"Invalid regex expression: {key} - {value}")
                logger.error(str(e))
        return regex_expressions

    def label_row(self, row) -> int:
        src = row[0]
        exec = row[2]
        dst = row[3]
        path = row[7]
        ip = row[8]
        port = row[9]
        timestamp = row[6]
        for i in range(len(self.gt_start_times)):
            if self.gt_start_times[i] <= int(timestamp) <= self.gt_end_times[i]:
                if self.check_search_terms(exec, path, ip, port, i) == 0:
                    return 0
        return 1

    def check_search_terms(self, exec, path, ip, port, i) -> int:
        search_terms = self.gt_search_terms[i]
        for term in search_terms:
            if term in exec or term in path or term in ip or term in port:
                return 0
        return 1

    def tt_to_dygraph_and_tgbgraph(
        self, exec_path=False, default_location=True, classification=False
    ):
        """format: Each line has the following format: source_node, destination_node, timestamp, edge_label, comma-separated arrays of edge features.
        * Please note that if there is no edge label available,
        the edge_label column will be filled with 0s only for loading purpose; these labels are not used in the link prediction task.
        * The first line denotes the network format.
        * Edge features should include at least one feature. If there is no edge feature available, a 0 value is used for all the edges.
        """
        logger.info("Converting ThreatTrace data to DyGraph format")

        header_input = [
            "source_uuid",
            "source_type",
            "exec",
            "destination_uuid",
            "destination_type",
            "edge_type",
            "timestamp",
            "path",
            "ip",
            "port",
        ]
        header_output = [
            "source_node",
            "destination_node",
            "timestamp",
            "edge_label",
            "edge_features",
        ]

        output_file = open(self.outputpath + "cadets.csv", "w")
        output_file_normal = open(self.outputpath + "cadets_normal.csv", "w")

        output_file.write(",".join(header_output) + "\n")

        uuid_mapping = {}
        type_mapping = {}
        exec_mapping = {}
        edge_type_mapping = {}
        ip_mapping: dict = {}
        port_mapping: dict = {}
        path_mapping: dict = {}
        socket_mapping: dict = {}
        location_mapping: dict = {}

        with open(self.outputpath + "eventlist.txt", "r") as file:
            lines = file.readlines()

        last_ts = 0
        self.edge_label_true_count = 0
        self.edge_label_false_count = 0
        unique_rows = set()
        for line in lines:
            row = line.strip().split(",")

            # skip some Events

            if row[5] not in self.keep_events:
                continue

            if row[2] in self.skip_execs:
                self.no_not_events += 1
                continue

            # Mapping for UUIDs
            source_uuid = row[0]
            destination_uuid = row[3]
            if source_uuid not in uuid_mapping:
                uuid_mapping[source_uuid] = len(uuid_mapping)
            if destination_uuid not in uuid_mapping:
                uuid_mapping[destination_uuid] = len(uuid_mapping)

            # Mapping for source_type and destination_type
            source_type = row[1]
            destination_type = row[4]
            if source_type not in type_mapping:
                type_mapping[source_type] = len(type_mapping)
            if destination_type not in type_mapping:
                type_mapping[destination_type] = len(type_mapping)

            exec = row[2]
            if exec not in exec_mapping:
                exec_mapping[exec] = len(exec_mapping)

            edge_type = row[5]
            if edge_type not in edge_type_mapping:
                edge_type_mapping[edge_type] = len(edge_type_mapping)

            path_type = row[7]
            if path_type not in path_mapping:
                path_mapping[path_type] = len(path_mapping)

            ip = row[8]
            if ip not in ip_mapping:
                ip_mapping[ip] = len(ip_mapping)

            port = row[9]
            if port not in port_mapping:
                port_mapping[port] = len(port_mapping)

            socket = row[7] + row[8]
            if socket not in socket_mapping:
                socket_mapping[socket] = len(socket_mapping)

            location = row[7] + row[8] + row[9]
            if location not in location_mapping:
                location_mapping[location] = len(location_mapping)

            if exec_path == False:
                feature_array = [
                    type_mapping[source_type],
                    exec_mapping[exec],
                    type_mapping[destination_type],
                    edge_type_mapping[edge_type],
                ]
            else:
                feature_array = [
                    type_mapping[source_type],
                    type_mapping[destination_type],
                    edge_type_mapping[edge_type],
                ]
                feature_array_normal = [source_type, destination_type, edge_type]

            timestamp = row[6]

            if last_ts > int(timestamp):
                logger.warning(
                    f"Timestamps not in ascending order: {last_ts} > {timestamp}"
                )
            last_ts = int(timestamp)

            edge_label = self.label_row(row)
            if edge_label == 0:
                self.edge_label_true_count += 1
            else:
                self.edge_label_false_count += 1

            if default_location == False and (
                location == " null null null" or location == " <unknown> null null"
            ):
                continue
            else:
                if exec_path == False:
                    output_file.write(
                        f"{uuid_mapping[source_uuid]},{uuid_mapping[destination_uuid]},{timestamp},{edge_label},{','.join(map(str, feature_array))}\n"
                    )
                else:
                    output_file.write(
                        f"{exec_mapping[exec]},{location_mapping[location]},{timestamp},{edge_label},{','.join(map(str, feature_array))}\n"
                    )
                    output_file_normal.write(
                        f"{exec},{location},{timestamp},{edge_label},{','.join(map(str, feature_array_normal))}\n"
                    )

        logger.info(f"Some Statistics: ")
        logger.info(f"  Number of nodes: {len(uuid_mapping)}")
        logger.info(f"  Number of edge types: {len(edge_type_mapping)}")
        logger.info(f"  Number of exec types: {len(exec_mapping)}")
        logger.info(f"  Number of source and destination types: {len(type_mapping)}")
        logger.info(f"  Number of edges: {len(lines)}")
        logger.info(f"  Number of Anomal edge labels: {self.edge_label_true_count}")
        logger.info(f"  Number of Normal edge labels: {self.edge_label_false_count}")

        output_file.close()
        output_file_normal.close()
        logger.info(f"Output file closed: {self.outputpath + 'cadets.csv'}")

    def to_unique_edgelist(self):
        output_file = self.outputpath + "cadets.csv"
        output_file_normal = self.outputpath + "cadets_normal.csv"
        header = [
            "source_node",
            "destination_node",
            "timestamp",
            "edge_label",
            "source_type",
            "destination_type",
            "edge_type",
        ]
        df_output = pd.read_csv(output_file, names=header)
        df_output_normal = pd.read_csv(output_file_normal, names=header)

        df_output["edge"] = df_output.apply(
            lambda x: f"{x['source_node']}_{x['destination_node']}_{x['destination_type']}_{x['edge_type']}",
            axis=1,
        )
        df_output_normal["edge"] = df_output_normal.apply(
            lambda x: f"{x['source_node']}_{x['destination_node']}_{x['destination_type']}_{x['edge_type']}",
            axis=1,
        )

        print(f"Len of df_output: {len(df_output)}")
        print(f"Len of df_output_normal: {len(df_output_normal)}")

        df_output = df_output.drop_duplicates(subset=["edge"], keep="first")
        df_output_normal = df_output_normal.drop_duplicates(
            subset=["edge"], keep="first"
        )

        df_output = df_output.drop(columns=["edge"])
        df_output_normal = df_output_normal.drop(columns=["edge"])
        print(f"Len of df_output after drop: {len(df_output)}")
        print(f"Len of df_output_normal after drop: {len(df_output_normal)}")

        output_file = self.outputpath + "cadets_unique.csv"
        output_file_normal = self.outputpath + "cadets_normal_unique.csv"
        df_output.to_csv(output_file, index=False, header=False)
        df_output_normal.to_csv(output_file_normal, index=False, header=False)

        df_output["timestamp"] = df_output.index

        logger.info(f"Unique edge list written to file: {output_file}")
        logger.info(f"Unique edge list written to file: {output_file_normal}")

    def generate_node_list_file(self):
        output_file = self.outputpath + "cadets_normal_unique_nodelist.csv"
        input_file = self.outputpath + "cadets_normal_unique.csv"
        header = ["src", "dst", "ts", "label", "src_type", "dst_type", "edge_type"]
        df = pd.read_csv(input_file, names=header)

        df_src = df[["src", "src_type"]]
        df_src.columns = ["node", "type"]
        df_dst = df[["dst", "dst_type"]]
        df_dst.columns = ["node", "type"]
        df_output = pd.concat([df_src, df_dst])
        df_output = df_output.drop_duplicates(subset=["node", "type"], keep="first")
        df_output.to_csv(output_file, index=False, header=False)
        logger.info(f"Node list written to file: {output_file}")

    def to_bidirectional_file(self, input_file):
        header = ["src", "dst", "ts", "label", "src_type", "dst_type", "edge_type"]
        df = pd.read_csv(input_file, names=header)
        if input_file.endswith("cadets_unique.csv"):
            df = df.iloc[1:]
        df_copy = df.copy()
        df_copy["src"] = df["dst"]
        df_copy["dst"] = df["src"]
        df_copy["src_type"] = df["dst_type"]
        df_copy["dst_type"] = df["src_type"]

        df_output = pd.concat([df, df_copy])
        output_file = input_file + "_bidirectional.csv"
        df_output = df_output.sort_values(by="ts")
        df_output.to_csv(output_file, index=False, header=False)

    def to_bidirectional(self):
        input_file = self.outputpath + "cadets_normal_unique.csv"
        self.to_bidirectional_file(input_file)
        logger.info(f"Bidirectional file written to: {input_file}_bidirectional.csv")
        input_file = self.outputpath + "cadets_unique.csv"
        self.to_bidirectional_file(input_file)
        logger.info(f"Bidirectional file written to: {input_file}_bidirectional.csv")


# if __name__ == "__main__":
#     import sys
#     from pathlib import Path
#     import logging
#     import os
#     import logging
#     import gdown
#     import tarfile
#     # Set system path to the folder where the notebook is saved
#     notebook_dir = Path().resolve()
#     sys.path.insert(0, str(notebook_dir))
#     parser = DarpaParser(dataset="cadets", config_type="all", uuid_writer=True)
#     # parser.parse_data_threatrace()
#     # parser.tt_to_dygraph_and_tgbgraph(
#     #     exec_path=True, default_location=False, classification=False
#     # )
#     # parser.to_unique_edgelist()
#     # parser.generate_node_list_file()
