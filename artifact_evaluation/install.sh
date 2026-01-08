
command -v uv &>/dev/null && echo "uv is not found in PATH, please install with https://docs.astral.sh/uv/getting-started/installation/"

# Compile libsmctrl
cd ../csrc
make config
make build

# Install Bullet
cd ..
conda create -n bullet python==3.12.9
conda activate bullet
uv pip install -e "python[all]"
uv pip install seaborn