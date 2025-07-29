conda create -n mvdream python=3.10 -y
conda activate mvdream 

pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install --upgrade --no-deps --force-reinstall xformers==0.0.23.post1 -i https://download.pytorch.org/whl/cu121
pip install -r ./stage_1_low_res/requirements.txt
pip install "git+https://github.com/bytedance/MVDream"
pip install "numpy<1.24"
pip install xatlas jaxtyping typeguard megfile distinctipy