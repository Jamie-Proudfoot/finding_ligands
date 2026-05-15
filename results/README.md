Folder containing csv filess of active learning results
Produced by the following:
1. (py3env) optimization.py (e.g. via sh submit_BO.sh)
(optional)
2. (py3env) summary_metrics.py
3. (py3env) write_summary.py

1. Files ending in .csv: contain steps-to-value results (average number of steps required to reach a certain activity threshold)

2. Files ending in ID.csv: contain 0-indexed integer ids of the compounds selected during screening (corresponding to their position in the csv files containing the data sets)

3. Files ending in SM.csv: contains summary metrics for the screening process, including average: steps-to-maximum, enrichment factor (EF, proportional to recall of top-1% most active compounds), and recall frequency of the maximum 
