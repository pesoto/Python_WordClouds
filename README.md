
# Word Clouds in Python using WordCram

This is a short tutorial for using WordCram, a Processing library, to generate word clouds from an LDA instance.
The LDA analysis will be done using Stephen Hansen's Topic Modelling library available at
https://github.com/sekhansen/text-mining-tutorial.

### Install Processing

First, Processing will need to be installed on your machine. The link for the download can be found at:
https://processing.org/download/?processing
    
After installation, make sure you know where the processing-java file was downloaded. This will be needed
for the Python module.

### Create LDA Instance

First, let's generate the ldaobj instance from the topicmodels.LDA class.


    """
    (c) 2014, Stephen Hansen, stephen.hansen@upf.edu
    
    Python script for tutorial illustrating collapsed Gibbs sampling for Latent Dirichlet Allocation.
    
    See explanation for commands on http://nbviewer.ipython.org/url/www.econ.upf.edu/~shansen/tutorial_notebook.ipynb.
    """
    
    import pandas as pd
    import topicmodels
    import sys
    from wordclouds import *
    
    topic_nums = 10
    
    ########## select data on which to run topic model #########
    
    data = pd.read_table("speech_data_extend.txt",encoding="utf-8")
    
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

    number of unique stems = 100
    number of total stems = 12436
    Iteration 1 of (collapsed) Gibbs sampling
    Iteration 2 of (collapsed) Gibbs sampling
    Iteration 3 of (collapsed) Gibbs sampling
    Iteration 4 of (collapsed) Gibbs sampling
    Iteration 5 of (collapsed) Gibbs sampling
    Iteration 6 of (collapsed) Gibbs sampling
    Iteration 7 of (collapsed) Gibbs sampling
    Iteration 8 of (collapsed) Gibbs sampling
    Iteration 9 of (collapsed) Gibbs sampling
    Iteration 10 of (collapsed) Gibbs sampling
    Iteration 11 of (collapsed) Gibbs sampling
    Iteration 12 of (collapsed) Gibbs sampling
    Iteration 13 of (collapsed) Gibbs sampling
    Iteration 14 of (collapsed) Gibbs sampling
    Iteration 15 of (collapsed) Gibbs sampling
    Iteration 16 of (collapsed) Gibbs sampling
    Iteration 17 of (collapsed) Gibbs sampling
    Iteration 18 of (collapsed) Gibbs sampling
    Iteration 19 of (collapsed) Gibbs sampling
    Iteration 20 of (collapsed) Gibbs sampling
    Iteration 21 of (collapsed) Gibbs sampling
    Iteration 22 of (collapsed) Gibbs sampling
    Iteration 23 of (collapsed) Gibbs sampling
    Iteration 24 of (collapsed) Gibbs sampling
    Iteration 25 of (collapsed) Gibbs sampling
    Iteration 26 of (collapsed) Gibbs sampling
    Iteration 27 of (collapsed) Gibbs sampling
    Iteration 28 of (collapsed) Gibbs sampling
    Iteration 29 of (collapsed) Gibbs sampling
    Iteration 30 of (collapsed) Gibbs sampling
    Iteration 31 of (collapsed) Gibbs sampling
    Iteration 32 of (collapsed) Gibbs sampling
    Iteration 33 of (collapsed) Gibbs sampling
    Iteration 34 of (collapsed) Gibbs sampling
    Iteration 35 of (collapsed) Gibbs sampling
    Iteration 36 of (collapsed) Gibbs sampling
    Iteration 37 of (collapsed) Gibbs sampling
    Iteration 38 of (collapsed) Gibbs sampling
    Iteration 39 of (collapsed) Gibbs sampling
    Iteration 40 of (collapsed) Gibbs sampling
    Iteration 41 of (collapsed) Gibbs sampling
    Iteration 42 of (collapsed) Gibbs sampling
    Iteration 43 of (collapsed) Gibbs sampling
    Iteration 44 of (collapsed) Gibbs sampling
    Iteration 45 of (collapsed) Gibbs sampling
    Iteration 46 of (collapsed) Gibbs sampling
    Iteration 47 of (collapsed) Gibbs sampling
    Iteration 48 of (collapsed) Gibbs sampling
    Iteration 49 of (collapsed) Gibbs sampling
    Iteration 50 of (collapsed) Gibbs sampling
    Iteration 51 of (collapsed) Gibbs sampling
    Iteration 52 of (collapsed) Gibbs sampling
    Iteration 53 of (collapsed) Gibbs sampling
    Iteration 54 of (collapsed) Gibbs sampling
    Iteration 55 of (collapsed) Gibbs sampling
    Iteration 56 of (collapsed) Gibbs sampling
    Iteration 57 of (collapsed) Gibbs sampling
    Iteration 58 of (collapsed) Gibbs sampling
    Iteration 59 of (collapsed) Gibbs sampling
    Iteration 60 of (collapsed) Gibbs sampling
    Iteration 61 of (collapsed) Gibbs sampling
    Iteration 62 of (collapsed) Gibbs sampling
    Iteration 63 of (collapsed) Gibbs sampling
    Iteration 64 of (collapsed) Gibbs sampling
    Iteration 65 of (collapsed) Gibbs sampling
    Iteration 66 of (collapsed) Gibbs sampling
    Iteration 67 of (collapsed) Gibbs sampling
    Iteration 68 of (collapsed) Gibbs sampling
    Iteration 69 of (collapsed) Gibbs sampling
    Iteration 70 of (collapsed) Gibbs sampling
    Iteration 71 of (collapsed) Gibbs sampling
    Iteration 72 of (collapsed) Gibbs sampling
    Iteration 73 of (collapsed) Gibbs sampling
    Iteration 74 of (collapsed) Gibbs sampling
    Iteration 75 of (collapsed) Gibbs sampling
    Iteration 76 of (collapsed) Gibbs sampling
    Iteration 77 of (collapsed) Gibbs sampling
    Iteration 78 of (collapsed) Gibbs sampling
    Iteration 79 of (collapsed) Gibbs sampling
    Iteration 80 of (collapsed) Gibbs sampling
    Iteration 81 of (collapsed) Gibbs sampling
    Iteration 82 of (collapsed) Gibbs sampling
    Iteration 83 of (collapsed) Gibbs sampling
    Iteration 84 of (collapsed) Gibbs sampling
    Iteration 85 of (collapsed) Gibbs sampling
    Iteration 86 of (collapsed) Gibbs sampling
    Iteration 87 of (collapsed) Gibbs sampling
    Iteration 88 of (collapsed) Gibbs sampling
    Iteration 89 of (collapsed) Gibbs sampling
    Iteration 90 of (collapsed) Gibbs sampling
    Iteration 91 of (collapsed) Gibbs sampling
    Iteration 92 of (collapsed) Gibbs sampling
    Iteration 93 of (collapsed) Gibbs sampling
    Iteration 94 of (collapsed) Gibbs sampling
    Iteration 95 of (collapsed) Gibbs sampling
    Iteration 96 of (collapsed) Gibbs sampling
    Iteration 97 of (collapsed) Gibbs sampling
    Iteration 98 of (collapsed) Gibbs sampling
    Iteration 99 of (collapsed) Gibbs sampling
    Iteration 100 of (collapsed) Gibbs sampling
    Iteration 101 of (collapsed) Gibbs sampling
    Iteration 102 of (collapsed) Gibbs sampling
    Iteration 103 of (collapsed) Gibbs sampling
    Iteration 104 of (collapsed) Gibbs sampling
    Iteration 105 of (collapsed) Gibbs sampling
    Iteration 106 of (collapsed) Gibbs sampling
    Iteration 107 of (collapsed) Gibbs sampling
    Iteration 108 of (collapsed) Gibbs sampling
    Iteration 109 of (collapsed) Gibbs sampling
    Iteration 110 of (collapsed) Gibbs sampling
    Iteration 111 of (collapsed) Gibbs sampling
    Iteration 112 of (collapsed) Gibbs sampling
    Iteration 113 of (collapsed) Gibbs sampling
    Iteration 114 of (collapsed) Gibbs sampling
    Iteration 115 of (collapsed) Gibbs sampling
    Iteration 116 of (collapsed) Gibbs sampling
    Iteration 117 of (collapsed) Gibbs sampling
    Iteration 118 of (collapsed) Gibbs sampling
    Iteration 119 of (collapsed) Gibbs sampling
    Iteration 120 of (collapsed) Gibbs sampling
    Iteration 121 of (collapsed) Gibbs sampling
    Iteration 122 of (collapsed) Gibbs sampling
    Iteration 123 of (collapsed) Gibbs sampling
    Iteration 124 of (collapsed) Gibbs sampling
    Iteration 125 of (collapsed) Gibbs sampling
    Iteration 126 of (collapsed) Gibbs sampling
    Iteration 127 of (collapsed) Gibbs sampling
    Iteration 128 of (collapsed) Gibbs sampling
    Iteration 129 of (collapsed) Gibbs sampling
    Iteration 130 of (collapsed) Gibbs sampling
    Iteration 131 of (collapsed) Gibbs sampling
    Iteration 132 of (collapsed) Gibbs sampling
    Iteration 133 of (collapsed) Gibbs sampling
    Iteration 134 of (collapsed) Gibbs sampling
    Iteration 135 of (collapsed) Gibbs sampling
    Iteration 136 of (collapsed) Gibbs sampling
    Iteration 137 of (collapsed) Gibbs sampling
    Iteration 138 of (collapsed) Gibbs sampling
    Iteration 139 of (collapsed) Gibbs sampling
    Iteration 140 of (collapsed) Gibbs sampling
    Iteration 141 of (collapsed) Gibbs sampling
    Iteration 142 of (collapsed) Gibbs sampling
    Iteration 143 of (collapsed) Gibbs sampling
    Iteration 144 of (collapsed) Gibbs sampling
    Iteration 145 of (collapsed) Gibbs sampling
    Iteration 146 of (collapsed) Gibbs sampling
    Iteration 147 of (collapsed) Gibbs sampling
    Iteration 148 of (collapsed) Gibbs sampling
    Iteration 149 of (collapsed) Gibbs sampling
    Iteration 150 of (collapsed) Gibbs sampling
    Iteration 151 of (collapsed) Gibbs sampling
    Iteration 152 of (collapsed) Gibbs sampling
    Iteration 153 of (collapsed) Gibbs sampling
    Iteration 154 of (collapsed) Gibbs sampling
    Iteration 155 of (collapsed) Gibbs sampling
    Iteration 156 of (collapsed) Gibbs sampling
    Iteration 157 of (collapsed) Gibbs sampling
    Iteration 158 of (collapsed) Gibbs sampling
    Iteration 159 of (collapsed) Gibbs sampling
    Iteration 160 of (collapsed) Gibbs sampling
    Iteration 161 of (collapsed) Gibbs sampling
    Iteration 162 of (collapsed) Gibbs sampling
    Iteration 163 of (collapsed) Gibbs sampling
    Iteration 164 of (collapsed) Gibbs sampling
    Iteration 165 of (collapsed) Gibbs sampling
    Iteration 166 of (collapsed) Gibbs sampling
    Iteration 167 of (collapsed) Gibbs sampling
    Iteration 168 of (collapsed) Gibbs sampling
    Iteration 169 of (collapsed) Gibbs sampling
    Iteration 170 of (collapsed) Gibbs sampling
    Iteration 171 of (collapsed) Gibbs sampling
    Iteration 172 of (collapsed) Gibbs sampling
    Iteration 173 of (collapsed) Gibbs sampling
    Iteration 174 of (collapsed) Gibbs sampling
    Iteration 175 of (collapsed) Gibbs sampling
    Iteration 176 of (collapsed) Gibbs sampling
    Iteration 177 of (collapsed) Gibbs sampling
    Iteration 178 of (collapsed) Gibbs sampling
    Iteration 179 of (collapsed) Gibbs sampling
    Iteration 180 of (collapsed) Gibbs sampling
    Iteration 181 of (collapsed) Gibbs sampling
    Iteration 182 of (collapsed) Gibbs sampling
    Iteration 183 of (collapsed) Gibbs sampling
    Iteration 184 of (collapsed) Gibbs sampling
    Iteration 185 of (collapsed) Gibbs sampling
    Iteration 186 of (collapsed) Gibbs sampling
    Iteration 187 of (collapsed) Gibbs sampling
    Iteration 188 of (collapsed) Gibbs sampling
    Iteration 189 of (collapsed) Gibbs sampling
    Iteration 190 of (collapsed) Gibbs sampling
    Iteration 191 of (collapsed) Gibbs sampling
    Iteration 192 of (collapsed) Gibbs sampling
    Iteration 193 of (collapsed) Gibbs sampling
    Iteration 194 of (collapsed) Gibbs sampling
    Iteration 195 of (collapsed) Gibbs sampling
    Iteration 196 of (collapsed) Gibbs sampling
    Iteration 197 of (collapsed) Gibbs sampling
    Iteration 198 of (collapsed) Gibbs sampling
    Iteration 199 of (collapsed) Gibbs sampling
    Iteration 200 of (collapsed) Gibbs sampling
    Iteration 1 of (collapsed) Gibbs sampling
    Iteration 2 of (collapsed) Gibbs sampling
    Iteration 3 of (collapsed) Gibbs sampling
    Iteration 4 of (collapsed) Gibbs sampling
    Iteration 5 of (collapsed) Gibbs sampling
    Iteration 6 of (collapsed) Gibbs sampling
    Iteration 7 of (collapsed) Gibbs sampling
    Iteration 8 of (collapsed) Gibbs sampling
    Iteration 9 of (collapsed) Gibbs sampling
    Iteration 10 of (collapsed) Gibbs sampling
    Iteration 11 of (collapsed) Gibbs sampling
    Iteration 12 of (collapsed) Gibbs sampling
    Iteration 13 of (collapsed) Gibbs sampling
    Iteration 14 of (collapsed) Gibbs sampling
    Iteration 15 of (collapsed) Gibbs sampling
    Iteration 16 of (collapsed) Gibbs sampling
    Iteration 17 of (collapsed) Gibbs sampling
    Iteration 18 of (collapsed) Gibbs sampling
    Iteration 19 of (collapsed) Gibbs sampling
    Iteration 20 of (collapsed) Gibbs sampling
    Iteration 21 of (collapsed) Gibbs sampling
    Iteration 22 of (collapsed) Gibbs sampling
    Iteration 23 of (collapsed) Gibbs sampling
    Iteration 24 of (collapsed) Gibbs sampling
    Iteration 25 of (collapsed) Gibbs sampling
    Iteration 26 of (collapsed) Gibbs sampling
    Iteration 27 of (collapsed) Gibbs sampling
    Iteration 28 of (collapsed) Gibbs sampling
    Iteration 29 of (collapsed) Gibbs sampling
    Iteration 30 of (collapsed) Gibbs sampling
    Iteration 31 of (collapsed) Gibbs sampling
    Iteration 32 of (collapsed) Gibbs sampling
    Iteration 33 of (collapsed) Gibbs sampling
    Iteration 34 of (collapsed) Gibbs sampling
    Iteration 35 of (collapsed) Gibbs sampling
    Iteration 36 of (collapsed) Gibbs sampling
    Iteration 37 of (collapsed) Gibbs sampling
    Iteration 38 of (collapsed) Gibbs sampling
    Iteration 39 of (collapsed) Gibbs sampling
    Iteration 40 of (collapsed) Gibbs sampling
    Iteration 41 of (collapsed) Gibbs sampling
    Iteration 42 of (collapsed) Gibbs sampling
    Iteration 43 of (collapsed) Gibbs sampling
    Iteration 44 of (collapsed) Gibbs sampling
    Iteration 45 of (collapsed) Gibbs sampling
    Iteration 46 of (collapsed) Gibbs sampling
    Iteration 47 of (collapsed) Gibbs sampling
    Iteration 48 of (collapsed) Gibbs sampling
    Iteration 49 of (collapsed) Gibbs sampling
    Iteration 50 of (collapsed) Gibbs sampling
    Iteration 51 of (collapsed) Gibbs sampling
    Iteration 52 of (collapsed) Gibbs sampling
    Iteration 53 of (collapsed) Gibbs sampling
    Iteration 54 of (collapsed) Gibbs sampling
    Iteration 55 of (collapsed) Gibbs sampling
    Iteration 56 of (collapsed) Gibbs sampling
    Iteration 57 of (collapsed) Gibbs sampling
    Iteration 58 of (collapsed) Gibbs sampling
    Iteration 59 of (collapsed) Gibbs sampling
    Iteration 60 of (collapsed) Gibbs sampling
    Iteration 61 of (collapsed) Gibbs sampling
    Iteration 62 of (collapsed) Gibbs sampling
    Iteration 63 of (collapsed) Gibbs sampling
    Iteration 64 of (collapsed) Gibbs sampling
    Iteration 65 of (collapsed) Gibbs sampling
    Iteration 66 of (collapsed) Gibbs sampling
    Iteration 67 of (collapsed) Gibbs sampling
    Iteration 68 of (collapsed) Gibbs sampling
    Iteration 69 of (collapsed) Gibbs sampling
    Iteration 70 of (collapsed) Gibbs sampling
    Iteration 71 of (collapsed) Gibbs sampling
    Iteration 72 of (collapsed) Gibbs sampling
    Iteration 73 of (collapsed) Gibbs sampling
    Iteration 74 of (collapsed) Gibbs sampling
    Iteration 75 of (collapsed) Gibbs sampling
    Iteration 76 of (collapsed) Gibbs sampling
    Iteration 77 of (collapsed) Gibbs sampling
    Iteration 78 of (collapsed) Gibbs sampling
    Iteration 79 of (collapsed) Gibbs sampling
    Iteration 80 of (collapsed) Gibbs sampling
    Iteration 81 of (collapsed) Gibbs sampling
    Iteration 82 of (collapsed) Gibbs sampling
    Iteration 83 of (collapsed) Gibbs sampling
    Iteration 84 of (collapsed) Gibbs sampling
    Iteration 85 of (collapsed) Gibbs sampling
    Iteration 86 of (collapsed) Gibbs sampling
    Iteration 87 of (collapsed) Gibbs sampling
    Iteration 88 of (collapsed) Gibbs sampling
    Iteration 89 of (collapsed) Gibbs sampling
    Iteration 90 of (collapsed) Gibbs sampling
    Iteration 91 of (collapsed) Gibbs sampling
    Iteration 92 of (collapsed) Gibbs sampling
    Iteration 93 of (collapsed) Gibbs sampling
    Iteration 94 of (collapsed) Gibbs sampling
    Iteration 95 of (collapsed) Gibbs sampling
    Iteration 96 of (collapsed) Gibbs sampling
    Iteration 97 of (collapsed) Gibbs sampling
    Iteration 98 of (collapsed) Gibbs sampling
    Iteration 99 of (collapsed) Gibbs sampling
    Iteration 100 of (collapsed) Gibbs sampling
    Iteration 101 of (collapsed) Gibbs sampling
    Iteration 102 of (collapsed) Gibbs sampling
    Iteration 103 of (collapsed) Gibbs sampling
    Iteration 104 of (collapsed) Gibbs sampling
    Iteration 105 of (collapsed) Gibbs sampling
    Iteration 106 of (collapsed) Gibbs sampling
    Iteration 107 of (collapsed) Gibbs sampling
    Iteration 108 of (collapsed) Gibbs sampling
    Iteration 109 of (collapsed) Gibbs sampling
    Iteration 110 of (collapsed) Gibbs sampling
    Iteration 111 of (collapsed) Gibbs sampling
    Iteration 112 of (collapsed) Gibbs sampling
    Iteration 113 of (collapsed) Gibbs sampling
    Iteration 114 of (collapsed) Gibbs sampling
    Iteration 115 of (collapsed) Gibbs sampling
    Iteration 116 of (collapsed) Gibbs sampling
    Iteration 117 of (collapsed) Gibbs sampling
    Iteration 118 of (collapsed) Gibbs sampling
    Iteration 119 of (collapsed) Gibbs sampling
    Iteration 120 of (collapsed) Gibbs sampling
    Iteration 121 of (collapsed) Gibbs sampling
    Iteration 122 of (collapsed) Gibbs sampling
    Iteration 123 of (collapsed) Gibbs sampling
    Iteration 124 of (collapsed) Gibbs sampling
    Iteration 125 of (collapsed) Gibbs sampling
    Iteration 126 of (collapsed) Gibbs sampling
    Iteration 127 of (collapsed) Gibbs sampling
    Iteration 128 of (collapsed) Gibbs sampling
    Iteration 129 of (collapsed) Gibbs sampling
    Iteration 130 of (collapsed) Gibbs sampling
    Iteration 131 of (collapsed) Gibbs sampling
    Iteration 132 of (collapsed) Gibbs sampling
    Iteration 133 of (collapsed) Gibbs sampling
    Iteration 134 of (collapsed) Gibbs sampling
    Iteration 135 of (collapsed) Gibbs sampling
    Iteration 136 of (collapsed) Gibbs sampling
    Iteration 137 of (collapsed) Gibbs sampling
    Iteration 138 of (collapsed) Gibbs sampling
    Iteration 139 of (collapsed) Gibbs sampling
    Iteration 140 of (collapsed) Gibbs sampling
    Iteration 141 of (collapsed) Gibbs sampling
    Iteration 142 of (collapsed) Gibbs sampling
    Iteration 143 of (collapsed) Gibbs sampling
    Iteration 144 of (collapsed) Gibbs sampling
    Iteration 145 of (collapsed) Gibbs sampling
    Iteration 146 of (collapsed) Gibbs sampling
    Iteration 147 of (collapsed) Gibbs sampling
    Iteration 148 of (collapsed) Gibbs sampling
    Iteration 149 of (collapsed) Gibbs sampling
    Iteration 150 of (collapsed) Gibbs sampling
    Iteration 151 of (collapsed) Gibbs sampling
    Iteration 152 of (collapsed) Gibbs sampling
    Iteration 153 of (collapsed) Gibbs sampling
    Iteration 154 of (collapsed) Gibbs sampling
    Iteration 155 of (collapsed) Gibbs sampling
    Iteration 156 of (collapsed) Gibbs sampling
    Iteration 157 of (collapsed) Gibbs sampling
    Iteration 158 of (collapsed) Gibbs sampling
    Iteration 159 of (collapsed) Gibbs sampling
    Iteration 160 of (collapsed) Gibbs sampling
    Iteration 161 of (collapsed) Gibbs sampling
    Iteration 162 of (collapsed) Gibbs sampling
    Iteration 163 of (collapsed) Gibbs sampling
    Iteration 164 of (collapsed) Gibbs sampling
    Iteration 165 of (collapsed) Gibbs sampling
    Iteration 166 of (collapsed) Gibbs sampling
    Iteration 167 of (collapsed) Gibbs sampling
    Iteration 168 of (collapsed) Gibbs sampling
    Iteration 169 of (collapsed) Gibbs sampling
    Iteration 170 of (collapsed) Gibbs sampling
    Iteration 171 of (collapsed) Gibbs sampling
    Iteration 172 of (collapsed) Gibbs sampling
    Iteration 173 of (collapsed) Gibbs sampling
    Iteration 174 of (collapsed) Gibbs sampling
    Iteration 175 of (collapsed) Gibbs sampling
    Iteration 176 of (collapsed) Gibbs sampling
    Iteration 177 of (collapsed) Gibbs sampling
    Iteration 178 of (collapsed) Gibbs sampling
    Iteration 179 of (collapsed) Gibbs sampling
    Iteration 180 of (collapsed) Gibbs sampling
    Iteration 181 of (collapsed) Gibbs sampling
    Iteration 182 of (collapsed) Gibbs sampling
    Iteration 183 of (collapsed) Gibbs sampling
    Iteration 184 of (collapsed) Gibbs sampling
    Iteration 185 of (collapsed) Gibbs sampling
    Iteration 186 of (collapsed) Gibbs sampling
    Iteration 187 of (collapsed) Gibbs sampling
    Iteration 188 of (collapsed) Gibbs sampling
    Iteration 189 of (collapsed) Gibbs sampling
    Iteration 190 of (collapsed) Gibbs sampling
    Iteration 191 of (collapsed) Gibbs sampling
    Iteration 192 of (collapsed) Gibbs sampling
    Iteration 193 of (collapsed) Gibbs sampling
    Iteration 194 of (collapsed) Gibbs sampling
    Iteration 195 of (collapsed) Gibbs sampling
    Iteration 196 of (collapsed) Gibbs sampling
    Iteration 197 of (collapsed) Gibbs sampling
    Iteration 198 of (collapsed) Gibbs sampling
    Iteration 199 of (collapsed) Gibbs sampling
    Iteration 200 of (collapsed) Gibbs sampling


###Make WordClouds

The function make_word_clouds contains 2 required arguments, the ldaobj instance and the path to Processing. 

The Processing path typically looks like "processing-2.2.1/processing-java". 

The other 2 arguments are to tweak the aesthetics of the PDF files generated for each word cloud.

<b>weight_interval</b> tells Python what size to make both the largest and smallest weighted words, and adjust the in-between weights accordingly. Typically, if there is a lot of dispersion in weights or frequencies, then weight_interval=(4,140) (DEFAULT) will result in nice looking charts. If the output is only showing a fraction of the words, then trying weight_interval=(4,50) will likely fix the issue. 

<b>max_words</b> is the maximum number of words you would like to appear in each word cloud. By default it is set to 250. Reducing this number may require adjusting the <b>weight_interval</b> variable accordingly so that all remaining words fit well in the image.


    make_word_clouds(ldaobj,"~/Downloads/processing-2.2.1/processing-java",weight_interval=(4,140))

That is it for now. For any questions or suggestions, please email me at paul.soto@upf.edu
