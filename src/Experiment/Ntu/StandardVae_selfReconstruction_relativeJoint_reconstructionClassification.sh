module load python-env/2.7.10-ml
export PYTHONPATH=$PYTHONPATH:$WRKDIR/DONOTREMOVE/git/FeatureLearningAndGestureRecognition/src
srun -N 1 -n 1 --mem-per-cpu=100000 -t72:00:00 --gres=gpu:p100:1 -p gpu python StandardVae_selfReconstruction_relativeJoint_reconstructionClassification.py --finetune $1