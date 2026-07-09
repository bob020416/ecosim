# Installation

ECoSim uses a thin wrapper around two heavy traffic-model backends. Use one
environment per backend.

## VBD Backend

```bash
cd third_party/VBD
conda env create -n ecosim-vbd -f environment.yml
conda activate ecosim-vbd
pip install git+https://github.com/waymo-research/waymax.git@main#egg=waymo-waymax
pip install "transformers==4.41.2"
pip install -e .
cd ../..
pip install -e .
```

VBD text inference uses DistilBERT through Hugging Face Transformers. Keep
Transformers on the 4.x line for the pinned VBD environment; Transformers 5.x
requires a newer PyTorch than this backend uses.

## CATK Backend

```bash
conda create -y -n ecosim-catk python=3.11.9
conda activate ecosim-catk
conda install -y -c conda-forge ffmpeg=4.3.2
pip install -r third_party/catk/install/requirements.txt
pip install torch_geometric
pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install --no-deps waymo-open-dataset-tf-2-12-0==1.6.4
pip install -e .
```

The PyTorch Geometric wheel URL must match the PyTorch and CUDA build in your
environment. The command above reflects the validated CATK setup used for this
release.

The runner uses the active Python interpreter, so activate the matching
environment before launching a config.
