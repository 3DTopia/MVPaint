conda create -n syncmvd python=3.9 -y
conda activate syncmvd

pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install xatlas accelerate configargparse ipython
pip install diffusers==0.30.1 transformers==4.36.0
conda install -c iopath iopath
conda install -c fvcore -c conda-forge fvcore
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

pip install cupy-cuda12x==12.2.0
pip install -U scikit-learn scipy matplotlib    
pip install opencv-python==4.12.0.88
pip install "numpy<1.27"
pip install open3d==0.14.1

pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.0_cu121.html
pip install einops trimesh megfile

### install nvdiffrast
mkdir thirdparty && cd thirdparty
git clone https://github.com/NVlabs/nvdiffrast.git
cd nvdiffrast
pip install .

pip install --upgrade --no-deps --force-reinstall xformers==0.0.23.post1 -i https://download.pytorch.org/whl/cu121