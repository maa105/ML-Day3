import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn


from sklearn.tree import export_graphviz
def plot(dtree):
    export_graphviz(dtree, out_file="tree.dot",  
                filled=True, rounded=True,
                special_characters=True)
    from subprocess import call
    call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=150'])

np.random.seed(2)

data=pd.read_csv('creditcard.csv')
#print(data.head())
#print(data.isna().any())

data.corrwith(data.Class).plot.bar(figsize=(20,10), title="Correlation with Class", fontsize=15,rot=45, grid=True)
#plt.show()

corr=data.corr()
print(corr.head())

sn.set(style="white")
mask=np.zeros_like(corr,dtype=bool)
mask[np.triu_indices_from(mask)]=True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(18, 15))

# Generate a custom diverging colormap
cmap = sn.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sn.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.show()