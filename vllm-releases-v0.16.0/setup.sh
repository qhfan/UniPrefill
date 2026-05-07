# apt install -y tmux
python use_existing_torch.py
pip install -r requirements/build.txt
SETUPTOOLS_SCM_PRETEND_VERSION=0.16.0 VLLM_USE_PRECOMPILED=1 pip install -v -e .
# SETUPTOOLS_SCM_PRETEND_VERSION=0.16.0 pip install -v --no-build-isolation -e .
