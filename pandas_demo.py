import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#用列表生成Series
s = pd.Series([1,3,5,np.nan,6,8])
print(s)
#用Series、字典、对象生成DataFrame
df2 = pd.DataFrame({
    'A':1,
    'B':pd.Timestamp('20130102'),
    'C':pd.Series(1,index=list(range(4)),dtype='float32')
})

df2.plot()
plt.show()