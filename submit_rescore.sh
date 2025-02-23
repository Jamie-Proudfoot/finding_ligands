eval "$(conda shell.bash hook)";
conda activate DXGB;
tsp bash -c "python -u gen_delta.py > gen_delta.out 2> gen_delta.err"
