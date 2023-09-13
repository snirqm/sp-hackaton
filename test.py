
#%%
import numpy as np
import sp
from sp.sp import get_data_np
arr = np.array([0]*1000, dtype=np.float64)
# check how long on average it takes to run the function
%timeit get_data_np(arr)

# %%
print(arr)