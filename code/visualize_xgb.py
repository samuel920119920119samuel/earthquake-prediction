# plot decision tree
from numpy import loadtxt
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_tree
import matplotlib.pyplot as plt
import matplotlib

bst = xgb.Booster({'nthread': 4})  # init model
bst.load_model('xgb_md.model')  # load data

#xgb.plot_tree(bst)#,render=False)
#fig = matplotlib.pyplot.gcf()
##fig.set_size_inches(150, 1000)
#fig.savefig('tree.png',dpi=5000)

# plot single tree
plot_tree(bst)
#plt.savefig("temp.pdf")
plt.savefig('books_read.png',dpi=5000)
plt.show()
