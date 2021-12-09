#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH -c16
#SBATCH --job-name="yoloD2"
#SBATCH --output=out/yolorD2.out
#SBATCH --mail-user=asbjotof@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --constraint=A100

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"


module load Python/3.8.6-GCCcore-10.2.0

cd yolorPaper/yolor
#cd yolor

#python train.py --batch-size 8 --data data/ownData.yaml --cfg cfg/yolor_p6.cfg --epochs 10  --weights yolor_p6.pt --name yolorLiten --hyp data/hyp.finetune.1280.yaml
#python -m torch.distributed.launch --nproc_per_node 2 --master_port 9527 train.py --batch-size 8 --data data/ownData.yaml --cfg cfg/yolor_p6.cfg --weights '' --device 0,1 --sync-bn --name yolorP --hyp data/hyp.finetune.1280.yaml --epochs 100
python -m torch.distributed.launch --nproc_per_node 8 --master_port 9527 train.py --batch-size 16 --data data/ownData.yaml --cfg models/yolor-d6.yaml --weights '' --device 0,1,2,3,4,5,6,7 --sync-bn --name yolorD --hyp data/hyp.finetune.1280.yaml --epochs 200

