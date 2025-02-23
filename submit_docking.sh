eval "$(conda shell.bash hook)";
conda activate py3env;
tsp bash -c "python -u gen_dockstring.py > gen_dockstring.out 2> gen_dockstring.err"
