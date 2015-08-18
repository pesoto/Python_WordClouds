"""
(c) 2014, Stephen Hansen, stephen.hansen@upf.edu

Python script for tutorial illustrating collapsed Gibbs sampling for Latent Dirichlet Allocation.

See explanation for commands on http://nbviewer.ipython.org/url/www.econ.upf.edu/~shansen/tutorial_notebook.ipynb.
"""

import pandas as pd
import topicmodels
import sys
from wordclouds import *

topic_nums = eval(sys.argv[1])

########## select data on which to run topic model #########

data = pd.read_table("speech_data_extend.txt",encoding="utf-8")
#data = data.ix[0:500]
#data = data[data.year >= 1947]

########## clean documents #########

docsobj = topicmodels.RawDocs(data.speech, "stopwords.txt")
docsobj.token_clean(1)
docsobj.stopword_remove("tokens")
docsobj.stem()
docsobj.tf_idf("stems")
docsobj.stopword_remove("stems",100)

all_stems = [s for d in docsobj.stems for s in d]
print("number of unique stems = %d" % len(set(all_stems)))
print("number of total stems = %d" % len(all_stems))

########## estimate topic model #########

ldaobj = topicmodels.LDA(docsobj.stems,topic_nums)

ldaobj.sample(0,20,10)
ldaobj.sample(0,20,10)

ldaobj.samples_keep(4)
ldaobj.topic_content(20)

# Choose how many of the last chains to keep - here it is 4
ldaobj.samples_keep(4)

make_word_clouds(ldaobj.tt_avg(),ldaobj.tokey_key,"~/Downloads/processing-2.2.1/processing-java",weight_interval=(4,140))