# conda update -n base conda --yes
# conda update --all --yes

# conda create --name spotscape python=3.9.7 --yes
# conda activate spotscape

conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia --yes
conda install -c conda-forge scanpy python-igraph leidenalg --yes
conda install pyg==2.4.0 -c pyg --yes
pip install pyg_lib==0.3.1 torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html 
conda install wandb -c conda-forge --yes
conda install -c pytorch faiss-cpu=1.7.4 --yes