#!/bin/bash
#SBATCH --job-name=st_0.0001_0.00_osl_0.01
#SBATCH --open-mode=append
#SBATCH --output=/scratch/ec2684/research/consistency-lm/osl_st_out/%A_%a_%j_%x.out
#SBATCH --error=/scratch/ec2684/research/consistency-lm/osl_st_out/%A_%a_%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtx8000:1
#SBATCH -c 4
#SBATCH --array=0-9
#SBATCH --mail-user=<ec2684@nyu.edu>
#SBATCH --mail-type=ALL
#SBATCH --signal=USR1@60
term_handler () {
    # catch and ignore TERM. we get multiple terms during shutdown, so best
    # to just do nothing
    # but still keep going with the python process
    wait "$CHILD"
}

usr1_handler () {
    echo "SLURM signaling preemption/times up (SLURM_PROCID $SLURM_PROCID)." 
    kill -s INT "$CHILD"  # send ctrl-c to python
    if {SHOULD_REQUEUE} && [ "$SLURM_PROCID" -eq "0" ]; then
        echo "Waiting 5s and resubmitting..."
        sleep 5
        echo "Resubmitting..."
        scontrol requeue $SLURM_JOB_ID
    fi
    wait "$CHILD"
}

trap 'usr1_handler' USR1
trap 'term_handler' TERM
singularity exec --nv --overlay $SCRATCH/singularity/overlay-25GB-500K-00.ext3:ro /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif  /bin/bash -c "
source /ext3/env.sh
conda activate
python3 -u /scratch/ec2684/research/consistency-lm/self_terminating/base/train.py --model-load-dir /scratch/ec2684/research/consistency-lm/runs/wikitext2_sentencized_st_lstm/0108_1954/wikitext2_sentencized_st_lstm____st-epsilon=0.0001/wikitextv2 --include-date --loss-type osl  --oversmoothing-weight 0.0${SLURM_ARRAY_TASK_ID}  --st-epsilon 0.0 --overwrite-strlm 1 --seed 0 --save-base-dir /scratch/ec2684/research/consistency-lm/runs/osl_st/0.0001/0.0${SLURM_ARRAY_TASK_ID}
"
