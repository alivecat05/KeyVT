### 1. **Install the inference package:**
```bash
# following the project llava
conda create -n keyvt python=3.10 -y
conda activate keyvt
pip install --upgrade pip
pip install -e ".[train]"
```

### 2. **Install the other package:**
```bash
pip install easydict flash-attn==2.5.7
```

### 3. **KeyV Geometry aware view sampling**
```bash
# Get camera parameters
python3 KeyV/fastvggt.py
# Divide the scene
python3 KeyV/cal_bias.py
# Allocate views for each segment
python3 KeyV/select_frames.py
```
### 4. **KeyT OT-based key tokens selection**
```bash
#Go to 
/ScanQA_SQA/cdviews/KeyT.py
```
### 5. **Evaluation**
