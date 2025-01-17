#!/bin/sh

#SBATCH --job-name=emo
#SBATCH --time=24:00:00
##SBATCH --partition=compute
##SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --mem=64GB 
#SBATCH --cpus-per-task=4
#SBATCH --output=./logs/slurm-%A_%a.out
#SBATCH --account=plaza

# activate conda environment
# module load miniconda3
# source /home/Plaza/.bashrc
# conda activate emotion

# check python version
python3 --version

MODEL_NAME='gpt-4o'
TESTSET="raw_data"
PROMPT_VERSION="p1"
# PERSONA='a woman' # a woman, a man, a non-binary person #mikel 1019
# PERSONA='None

echo $PROMPT_VERSION

python ./2_get_completions_gpt4.py \
    --model_name_or_path $MODEL_NAME \
    --test_data_input_path ./Multitarget-CONAN.csv \
    --prompt_version $PROMPT_VERSION \
    # --persona "$PERSONA" \ #mikel 1019
    -- n_test_samples 5002 \
    --test_data_output_path ./12172024/gpt_prompt.csv
    # --test_data_output_path ./evaluation/data/model_completions/$MODEL_NAME/$PROMPT_VERSION/$PERSONA/$TESTSET.csv
