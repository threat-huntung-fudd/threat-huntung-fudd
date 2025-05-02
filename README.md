# `Fudd`

Welcome to the repository for **Fudd**, a Threat Hunting Framework utilizing graph-based anomaly detection on log data.

## Overview

This repository provides the code associated with the paper:

**_Fudd: Threat Hunting Framework Utilizing Graph-based Anomaly Detection on Log Data_**

The Repository encompasses the following components:
- **Data Preprocessing**: Preparing log data for analysis. All preprocessing steps for the cadets and trace dataset. 
- **Anomaly Detection**: Identifying unusual patterns through TGN and DyGFormer.
- **Anomaly Investigation**: Exploring detected anomalies in a graph database. With the queries we use. The Memgraph environment and the data upload are also provided. 


---

## 🙏 Acknowledgements

This codebase is built upon the following excellent repositories:

- [CTDG-link-anomaly-detection](https://github.com/timpostuvan/CTDG-link-anomaly-detection)
- [TGB](https://github.com/shenyangHuang/TGB)

---

## ⚙️ Installation

```bash
conda env create -f environment.yml
```
```bash
cd gnn 
git clone https://github.com/timpostuvan/TGB-link-anomaly-detection.git
cd TGB-link-anomaly-detection
pip install -e .
```
---

## 📊 Preprocessing
- Start the e.g. cadets.ipynb 
---
### 🧠 GNN Model Training
```bash
python train_darpa_cadets.py 
```
---
### 📈 Anomaly Detection 
```bash
python result_analysis_cadets.py
```
---
###🧪 Memgraph Hunting \& Visualization
```bash
cd memgraph-platform
docker-compose up -d
```
```bash
python darpa_upload_cadets.py
```
	•	Open http://localhost:3000 in your browser.
	•	Load the visual styling by uploading memgraph_style.txt.
	•	Run the provided Cypher queries to explore detected anomalies.
---
<!---
threat-huntung-fudd/threat-huntung-fudd is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
