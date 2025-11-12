import os
import glob
import math

import numpy as np
import pandas as pd
from tqdm import tqdm
from natsort import natsorted

dir = "results/BRR_greedy"

with open(os.path.join(dir,"summary.txt"),"w+") as f:
    for file in tqdm(natsorted(glob.glob(os.path.join(dir,"*SM.csv")))):
        filename = os.path.basename(os.path.normpath(file))
        job = filename[:-7]
        sm_df = pd.read_csv(file)
        ms = str(np.round(sm_df.mean_steps.values[0],0))
        sds = str(np.round(sm_df.std_steps.values[0],0))
        mef = str(np.round(sm_df.mean_EF.values[0],2))
        sdef = str(np.round(sm_df.std_EF.values[0],2))
        rec = str(int(np.round(sm_df.recall.values[0],2)*100))
        f.write(" ".join([job, ms, sds, mef, sdef, rec, "\n"]))


