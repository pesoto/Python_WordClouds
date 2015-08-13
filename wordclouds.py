import numpy as np
from subprocess import call
import os 
import shutil

##### THIS FILE CONTAINS THE FUNCTION FOR CREATING WORDCLOUDS USING PROCESSING


def make_word_clouds(ldaobj,proc_path,weight_interval=(4,140),max_words=250):
	""""
	ldaobj = instance of topicmodels.LDA; should contain at least 1 sample
	proc_path =  path to processing-java file 
	weight_interval = a tuple (minSize,maxSize) where minSize is the size to draw a Word of weight 0
												and maxSize is the size to draw a Word of weight 1

	NOTE: PROCESSING should be installed on your machine
			https://processing.org/download/

	INPUT:
			make_word_clouds(ldaobj,'~/Downloads/processing-2.2.1/processing-java')
	OUTPUT:
			TopicCloud0.pdf, TopicCloud1.pdf, ....
	"""
	# MAKE CLOUD
	TT = ldaobj.tt_avg()

	# NEED TO LOOP OVER TOPICS

	topics = range(0,ldaobj.tt.shape[1])

	for t in topics:
		topic = t
		proportions = 1000*np.transpose(TT)
		proportions = np.around(proportions)
		proportions = proportions.astype(int)

		# cleanedtokens = NEED TO FLIP ldaobj.token_key['inflat'] - mapping to number (which is row of matrix)
		cleantokens = {v: k for k, v in ldaobj.token_key.items()}

	    # CREATE TEXT FILE (.PDE) FOR PROCESSING PROGRAM 
		word_cloud = """import processing.pdf.*;
						import wordcram.*;
						PFont georgia = createFont("serif", 1);
						String outstart= "Cloud_Images/";
						String outfile = outstart + "TopicCloud.pdf";
						"""
		new_word_string = ''

		# ADD THE FREQUENCIES FOR EACH WORD
		freq = [(cleantokens[w],count) for w,count in zip(cleantokens,proportions[topic,:]) if count > 0]
		for i,fr in enumerate(freq):
				word_cloud += ("Word a%d = new Word(\"%s\", %d);\n" % (i,fr[0],fr[1]))
				new_word_string += ("a%d," % i)

		# APPEND FOOTER TO .PDE FILE
		word_cloud += """void setup() {
						  size(500, 300, PDF, outfile);
						  background(255);
						  
						  new WordCram(this)
						.fromWords(new Word[] {
							%s })
						.withColors(#000000, #707070)
						    .sizedByWeight%s
						    .minShapeSize(1)
						    .angledAt(0)
						    .maxNumberOfWordsToDraw(%s)
						    .withFont("serif")
						    .withWordPadding(3)
						    .drawAll();
						  
						  exit();
						}
						""" % (new_word_string,str(weight_interval),max_words)

		# CREATE FOLDER FOR .PDE FILE 
		if not os.path.exists("Processing_Cloud_File"):
			os.makedirs("Processing_Cloud_File")

		# WRITE .PDE FILE
		text_file = open('Processing_Cloud_File//Processing_Cloud_File.pde','w')
		text_file.write(word_cloud)
		text_file.close()


		# SEND COMMAND TO PROCESSING
		os.system(proc_path+" --run --sketch=Processing_Cloud_File --output=Processing_Cloud_File1")
		os.rename("Processing_Cloud_File/Cloud_Images/TopicCloud.pdf","TopicCloud"+str(t)+".pdf")

		# DELETE EXCESSIVE FILES
		shutil.rmtree('Processing_Cloud_File')
		shutil.rmtree('Processing_Cloud_File1')
