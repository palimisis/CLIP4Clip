#/bin/bash

# Instructions: Call without arguments to start training from the
# beginning. Call with a single argument which will be the checkpoint
# file to start from a previously saved checkpoint.
#
# If checkfile.chk is the checkpoint filename, then number of
# completed epochs is stored in a file named checkfile.numepochs
#
# Please define the `EXP_NAME` variable in order to correspond
# to the name of the running experiment.
#
# The N_STEPS variable is the number of times the experiment will be run.
# The EPOCHS_PER_STEP is the epochs that the model will be trained in each
# experiment. Both the EPOCHS_PER_STEP var is passed to the .py script
# as an argument. The example .py script handles the rest.
#
# For example: If I want to train my model for 30 epochs and
# train my model for 3 epochs each run, I will set:
# N_STEPS = 10 , EPOCHS_PER_STEP=3


EXP_NAME="CLIP4clip"
TOTAL_EPOCHS=10

CHK_PREFIX="/home/$(whoami)/experiments/${EXP_NAME}"
DATE=`date +"%s"`
LOGDIR="${CHK_PREFIX}/${DATE}/logs"
mkdir -p "${CHK_PREFIX}/${DATE}"
mkdir -p "${LOGDIR}"
CHK_NAME="${EXP_NAME}.chk"
jid=${EXP_NAME}_0
STDOUT="${LOGDIR}/$jid"
CUR_EPOCH=1

CKPT_PATH="${CHK_PREFIX}/${DATE}/ckpts/ckpt_msvd_retrieval_looseType"

sbatch -J $jid -o "${STDOUT}.out" -e "${STDOUT}.err" /home/it21902/CLIP4Clip/scripts/train_msvd.sh \
                                                                            --output_dir ${CKPT_PATH}

for ((i=1; i<${TOTAL_EPOCHS}; i++)); do
    let d=$i-1
    jid=${EXP_NAME}_$i
    STDOUT="${LOGDIR}/$jid"
    depends=$(squeue --noheader --format %i --name ${EXP_NAME}_${d})
    sbatch -J $jid -o "${STDOUT}.out" -e "${STDOUT}.err" -d afterany:${depends} /home/it21902/CLIP4Clip/scripts/train_msvd.sh \
                                                                                    --output_dir ${CKPT_PATH} \
                                                                                    --resume_model "${CKPT_PATH}\pytorch_model.bin.${d}"                                                                                    
done
