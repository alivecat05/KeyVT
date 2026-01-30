### 2. **Install the inference package:**
```bash
# following the project llava
conda create -n keyvt python=3.10 -y
conda activate keyvt
pip install --upgrade pip
pip install -e ".[train]"
```

### 3. **Install the other package:**
```bash
pip install easydict flash-attn==2.5.7