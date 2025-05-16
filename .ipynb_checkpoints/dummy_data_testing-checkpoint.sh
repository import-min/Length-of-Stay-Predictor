#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH	--mem-per-cpu=380000M
#SBATCH -p bii
#SBATCH -A nssac_students

# Initialize Conda and activate the environment
eval "$(conda shell.bash hook)"
if ! command -v conda &> /dev/null; then
    echo "Conda is not available. Please ensure Conda is installed and accessible."
    exit 1
fi

# Check if the 'lp' environment exists
if ! conda info --envs | grep -q "^lp "; then
    echo "Conda environment 'lp' does not exist. Please create it or check the environment name."
    exit 1
fi

# Set PYTHONPATH to recognize 'src' directory
export PYTHONPATH=$(pwd):$PYTHONPATH

# Load necessary modules and activate the environment
module load miniforge 
source activate lp

# Run Python script
echo "Running optimization stage..."
python3 src/optimization/optimize_nn.py || { echo "Optimization failed"; exit 1; }
echo -e "\nOptimization stage complete. Running training stage now..."

python3 src/training/train_nn.py || { echo "Training failed"; exit 1; }
echo -e "\nTraining stage complete. Running evaluation stage now..."

python3 src/evaluation/evaluate_nn.py || { echo "Evaluation failed"; exit 1; }
echo -e "\nPipeline successfully completed!"