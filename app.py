from flask import Flask,jsonify
from flask import render_template
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import csv
import pycountry
import threading
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from http.client import IncompleteRead
from flask import request

ckey = ''
csecret = ''
atoken = ''
asecret = ''

app = Flask(__name__)

good = ["sexy", "handsome", "beautiful", "funny", "happy", "winner" ,"good", "haha", "best", "lol", ":)", "best", "super", "amazing", "awesome", "cool", "genius", "top"
 			"recovered", "birth", "nice", "fun", "joy", "enjoy", "trust", "love" ]
bad = ["bad", "sad", "mad", "crack", "rain", "cheat", "hospital", "wounded", "dentist", "doctor", "crash", "death", "fat", "miss", "wrong", "ripped", "destroy", "rape",
		"sucide", "bombed", "died", "dead", "terrible", "horrible", "tough", "unhappy", "terrifying", "dissapoint", "fuck", "disgrace", "suck", ":(", "hate", "depressed", "angry", "shit", "crap", "nuts", "dirty", "ditch", "bitch"]
"""
bf=np.zeros(len(good))
gf=np.ones(len(bad))
f = np.concatenate((good,bad))
data = np.concatenate((gf,bf))
#print(data)
#print(f)

df = pd.read_csv('imdb_labelled.txt', delimiter="\t")
f = df['Phrase'].values
twenty_train = df['Sentiment'].values
#twenty_train = [0 if i<=4 else 1 for i in twenty_train ]
#print(twenty_train)
#df=pd.read_csv("train2.csv")
#f=df["SentimentText"].values
#twenty_train=df["Sentiment"].values
#twenty_train = data
count_vect = CountVectorizer()
f_counts = count_vect.fit_transform(f)

from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(f_counts)
f_tf = tf_transformer.transform(f_counts)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(f_counts)
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, twenty_train)

#print(X_train_tfidf.shape)
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, twenty_train)


from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer()), ('tfdidf', TfidfTransformer()), ('clf', MultinomialNB())])
text_clf = text_clf.fit(f, twenty_train)
from sklearn.linear_model import SGDClassifier
text_clf = Pipeline([('vect', CountVectorizer()), ('tfdidf', TfidfTransformer()), ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))])
text_clf.fit(f,twenty_train)
"""
""" --------------------------------"""
from colour import Color
import math
countries = {}
alpha_2 = [c.alpha_2 for c in list(pycountry.countries)]
color_array = list(Color("red").range_to("green",21))

def store(data):
	global i,g,b
	if("text" in data):
		#tweet = data.split(',"text":"')[1].split('","source')[0]
		# #print(len(tweet))
		# print(data)
		data = json.loads(data)
		#print(data["text"])
		if("place" in data.keys() and data["place"] != None ):
			x = data["place"]["country_code"]
			if( x in alpha_2):
				if(x not in countries.keys()):
					countries[x]= {"avg":[], "curr": []}

				#X_new_counts = tf_transformer.transform([ data["text"] ])
				#X_new_tfidf = tfidf_transformer.transform(X_new_counts)

				if(len(countries[x]["curr"]) >= 10 ):
					countries[x]["avg"].append(sum(countries[x]["curr"])/10)
					if(len(countries[x]["avg"]) > 10 ):
						countries[x]["avg"].pop(0)
					countries[x]["curr"] = []
				#print(data["text"] , ": ",text_clf.predict([data["text"]]))

				countries[x]["curr"].append(classify(data["text"].lower()))

def classify(text):
	g=0;
	b=0;
	for i in good:
		if(i in text):
			g += 1
	for j in bad:
		if(j in text):
			b += 1
	if(g>b):
		return 1
	if b>g :
		#print("\n",text,"\n")
		return -1
	return 0;

def mapToColor():
	count_color = {}

	print(countries)
	for country in countries.keys():
		if(len(countries[country]["avg"]) != 0):
			a = len(countries[country]["avg"])
			key = pycountry.countries.get(alpha_2=country).alpha_3
			count_color[key] = str(color_array[math.floor((sum(countries[country]["avg"]) / (1 if a == 0 else a)+1)*10)])
	return count_color

class listener(StreamListener):

	def on_data(self, data):
		store(data)
		return True

	def on_error(self, status):
	    print(status)
	    return True


auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)
twitterStream = Stream(auth, listener())

""" --------------------------------"""

class ThreadingExample(object):

	def __init__(self):
	    thread = threading.Thread(target=self.run, args=())
	    thread.daemon = True                  # Daemonize thread
	    thread.start()                        # Start the execution

	def run(self):
		while True:
			try:
				twitterStream.filter(locations = [-180,-90,180,90], languages=["en"])
			except IncompleteRead:
				print("IncompleteRead Ocuured\n\n")
				twitterStream.filter(locations = [-180,-90,180,90], languages=["en"])
				continue


"""----------------------------"""

@app.route('/colors')
def colors():
	colorDict = mapToColor()
	return jsonify(**colorDict)
@app.route('/')
def f1():
	example = ThreadingExample()
	return render_template('emomap.html')

# @app.route("/time_get", methods = ["POST","GET"])
# def time_get():
# 	return render_template("time_get.html")
#
# @app.route("/timeline_route", methods = ["POST","GET"])
# def modi_kun():
# 		if(request.method == "POST"):
# 			import os
# 			os.system("python2 gotm/Exporter.py --querysearch '" + request.form["key"]+"' --since '"+request.form["start_date"]+"' --until '"+request.form["end_date"]+"' --maxtweets 30")
# 			cnt = {}
# 			cnames = [c.name for c in list(pycountry.countries)]
# 			reader = csv.reader(open('output_got.csv'), delimiter=";")
# 			next(reader)
# 			for row in reader:
# 				for cn in cnames:
# 					if(cn.lower() in row[4].lower()):
# 						if(pycountry.countries.get(name=cn).alpha_3 not in cnt.keys()):
# 							cnt[pycountry.countries.get(name=cn).alpha_3] = 0
# 						cnt[pycountry.countries.get(name=cn).alpha_3] += 1
# 			cnt_color = {}
# 			for cn in cnt.keys():
# 				cnt_color[cn] = "rgba(255,0,0,"+str(cnt[cn]/sum(cnt.values()))+")"
# 			print(cnt)
# 			return jsonify(**cnt_color)


@app.route("/timeline", methods=["GET", "POST"])
def timeline():
	if(request.method == "POST"):
		import os
		k = request.json["key"]
		start_date = request.json["start_date"]
		end_date = request.json["end_date"]
		m = request.json["max"]
		qu = 'python2 gotm/Exporter.py --querysearch "'+k+'" --since '+start_date+" --until "+end_date+" --maxtweets "+str(m)
		print(request.json)
		os.system(qu)
		#os.system('python2 gotm/Exporter.py --querysearch "'+"Narcos"+'" --since '+"2016-09-20"+" --until "+"2016-11-20"+" --maxtweets 400")
		cnt = {}
		cnames = [c.name for c in list(pycountry.countries)]
		reader = csv.reader(open('output_got.csv'), delimiter=";")
		next(reader)
		clt = "USA"
		for row in reader:
			for cn in cnames:
				if(cn.lower() in row[4].lower()) or ( cn.lower() in row[5].lower() ):
					if(pycountry.countries.get(name=cn).alpha_3 not in cnt.keys()):
						cnt[pycountry.countries.get(name=cn).alpha_3] = 0
					else:
						if clt not in cnt.keys() and clt.lower() in row[4].lower() :
							cnt[clt] = 0
						if clt in cnt.keys() and clt.lower() in row[4].lower() :
							cnt[clt] += 1
						cnt[pycountry.countries.get(name=cn).alpha_3] += 1
		cnt_color = {}
		for cn in cnt.keys():
			cnt_color[cn] = "rgba(255,0,0,"+str(cnt[cn]/sum(cnt.values())+0.2)+")"
		print(cnt)
		return json.dumps(cnt_color, 200, {'ContentType':'application/json'} )
	return render_template("timeline.php")

app.run(debug=True)
