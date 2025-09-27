CONDA_ROOT=/share/apps/anaconda3/2021.05

# Source the conda.sh hook so that `conda activate` works in bash
if [ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]; then
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
else
  echo "ERROR: cannot find ${CONDA_ROOT}/etc/profile.d/conda.sh" >&2
  exit 1
fi

conda activate qutcc

export TMPDIR=/tmp
export XDG_RUNTIME_DIR=/tmp

python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.device_count())"