#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q gpua100
### -- set the job Name -- 
#BSUB -J motion-diffusion-prior
### -- ask for number of cores (default: 1) -- 
#BSUB -n 16
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=2:mode=exclusive_process"
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 4GB of memory per core/slot -- 
#BSUB -R "rusage[mem=32GB]"
### -- specify that we want the job to get killed if it exceeds 5 GB per core/slot -- 
##BSUB -M 5GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 8:00 
### -- set the email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u andersbthuesen@gmail.com
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o Output_%J.out 
#BSUB -e Output_%J.err 

# here follow the commands you want to execute with input.in as the input file
module load cuda/12.4.1 
export PATH="/zhome/87/0/137565/miniconda3/bin:$PATH"
. "/zhome/87/0/137565/miniconda3/etc/profile.d/conda.sh"
conda activate teton
python train.py
