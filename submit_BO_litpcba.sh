eval "$(conda shell.bash hook)";
conda activate py3env;
export OMP_NUM_THREADS=1;
export OPENBLAS_NUM_THREADS=1;
export MKL_NUM_THREADS=1;
export VECLIB_MAXIMUM_THREADS=1;
export NUMEXPR_NUM_THREADS=1;
nohup python -u optimization_litpcba.py > optimization_litpcba.out 2> optimization_litpcba.err &
