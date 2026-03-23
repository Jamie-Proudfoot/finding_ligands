eval "$(conda shell.bash hook)";
conda activate py3env;
export OMP_NUM_THREADS=1;
export OPENBLAS_NUM_THREADS=1;
export MKL_NUM_THREADS=1;
export VECLIB_MAXIMUM_THREADS=1;
export NUMEXPR_NUM_THREADS=1;
#tsp bash -c "python -u optimization.py > optimization.out 2> optimization.err" # if task-spooler is available
nohup python -u optimization.py > optimization.out 2> optimization.err &
