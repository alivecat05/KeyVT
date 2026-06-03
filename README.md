# Zero-Shot 3D Question Answering via Hierarchical View-to-Token Transportation

---

![Framework](framework_traj-1.png)

## Timeline & News

* **Jun 03, 2026** | Preprint available on [arXiv](https://arxiv.org/abs/2606.03100).
* **May 01, 2026** | Paper officially accepted by [ICML 2026](https://icml.cc/virtual/2026/poster/63796).
* **Jan 30, 2026** | Repository initialized.

---

## Installation

### 1. Environment Setup
Configure the core environment (compatible with LLaVA):
```bash
conda create -n keyvt python=3.10 -y
conda activate keyvt
pip install --upgrade pip
pip install -e ".[train]"
```

### 2. Dependencies
Install additional acceleration and configuration packages:
```bash
pip install easydict flash-attn==2.5.7
```

---

## Pipeline Execution

### 3. KeyV: Geometry-Aware View Sampling
Run the following scripts sequentially to process scene segmentation and view allocation:
```bash
# Extract camera parameters
python3 KeyV/fastvggt.py

# Partition the 3D scene
python3 KeyV/cal_bias.py

# Allocate views for each segment
python3 KeyV/select_frames.py
```

### 4. KeyT: OT-Based Token Selection
Execute the Optimal Transport (OT) based token selection module:
```bash
python3 ScanQA_SQA/cdviews/KeyT.py
```

---

## Evaluation

### 5. Benchmark Evaluation
Execute the corresponding shell scripts to evaluate performance on target benchmarks.

**ScanQA & SQA3D:**
```bash
cd ./geo/ScanQA_SQA/scripts
bash eval.sh
```

**VSI-bench:**
```bash
cd ./geo/vsi_test/scripts
bash evaluate_vsibench.sh
```
