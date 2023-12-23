# MegaBlocks

## Setup

### sub-module

clone Megatron-LM

```bash
git submodule update --init
```

### pip install

1. create virtual environment

    ```bash
    python -m venv .env
    source .env/bin/activate
    ```

2. update pip

      ```bash
      pip install --upgrade pip
      ```


3. install requirements

    ```bash
    pip install -r requirements.txt
    ```
    installing packages for Megatron-LM (ex: PyTorch)

    ```bash
    pip install ninja wheel packaging

    pip install numpy
    ```

    To install stanford-stk, you need to have PyTorch and numpy installed, as described in the README.
    ```bash
    pip install stanford-stk>=0.0.6
    pip install triton==2.1.0
    ```

4. install apex

    ```bash
    git clone git@github.com:NVIDIA/apex.git
    cd apex
    pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
    ```

5. install flash-attention

    ```bash
    pip install flash-attn --no-build-isolation
    ```

6. install packages for Megatron-LM

    ```bash
    pip install zarr
    pip install tensorstore
    ```
