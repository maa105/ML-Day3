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
