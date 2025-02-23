eval "$(conda shell.bash hook)";
conda activate py3env;
export OMP_NUM_THREADS=5;
export OPENBLAS_NUM_THREADS=5;
export MKL_NUM_THREADS=5;
export VECLIB_MAXIMUM_THREADS=5;
export NUMEXPR_NUM_THREADS=5;
#tsp bash -c "python -u active_learning_litpcba.py > active_learning_litpcba.out 2> active_learning_litpcba.err" # if task-spooler is available
nohup python -u active_learning_litpca.py > active_learning_litpcba.out 2> active_learning_litpcba.err &
