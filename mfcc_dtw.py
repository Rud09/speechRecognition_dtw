import librosa
from dtw import dtw
import IPython.display
import numpy as np
import speech_recognition as sr
import matplotlib.pyplot as plt
#%matplotlib inline


y1, sr1 = librosa.load('/Users/chris/Documents/audio_samples/elias_warping_cut.m4a')
y2, sr2 = librosa.load('/Users/chris/Documents/audio_samples/elias_leader_cut.m4a')
y3, sr3 = librosa.load('/Users/chris/Documents/audio_samples/chris_warping_cut.m4a')
y4, sr4 = librosa.load('/Users/chris/Documents/audio_samples/chris_follow_cut.m4a')
yX, srX = librosa.load('/Users/chris/Documents/audio_samples/full_sentence5.m4a')

from matplotlib import pyplot as plt
def show_image(dist,cost,path):
    plt.imshow(cost.T, origin='lower', interpolation='nearest')
    plt.plot(path[0], path[1], 'w')
    plt.xlim((-0.5, cost.shape[0]-0.5))
    plt.ylim((-0.5, cost.shape[1]-0.5))



#### 
import copy
def preprocess_mfcc(mfcc):
    mfcc_cp = copy.deepcopy(mfcc)
    for i in xrange(mfcc.shape[1]):
        mfcc_cp[:,i] = mfcc[:,i] - np.mean(mfcc[:,i])
        mfcc_cp[:,i] = mfcc_cp[:,i]/np.max(np.abs(mfcc_cp[:,i]))
    return mfcc_cp

mfccX1 = preprocess_mfcc(mfccX)
words = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila", "tree", "wow"]

y1, sr1 = librosa.load('/Users/chris/Documents/audio_samples/elias_mothers_milk_word.m4a')
y2, sr2 = librosa.load('/Users/chris/Documents/audio_samples/chris_mothers_milk_word.m4a')
y3, sr3 = librosa.load('/Users/chris/Documents/audio_samples/yaoquan_mothers_milk_word.m4a')
y4, sr4 = librosa.load('/Users/chris/Documents/audio_samples/chris_mothers_milk_word_slow.m4a')
yX, srX = librosa.load('/Users/chris/Documents/audio_samples/chris_mothers_milk_sentence_fast.m4a')

mfcc1 = librosa.feature.mfcc(y1, sr1)
mfcc2 = librosa.feature.mfcc(y2, sr2)
mfcc3 = librosa.feature.mfcc(y3, sr3)
mfcc4 = librosa.feature.mfcc(y4, sr4)
mfccX = librosa.feature.mfcc(yX, srX)

mfcc1 = preprocess_mfcc(mfcc1)
mfcc2 = preprocess_mfcc(mfcc2)
mfcc3 = preprocess_mfcc(mfcc3)
mfcc4 = preprocess_mfcc(mfcc4)
mfccX = preprocess_mfcc(mfccX)



speed = 0
dists = np.zeros(mfccX.shape[1]- window_size) + 1.5
window_size = (mfcc1.shape[1]+speed)
for i in range(mfccX.shape[1] - window_size - 1):

    mfcci = mfccX[:,i:i+window_size]

    dist3i, cost3i, path3i = dtw(mfcc1.T, mfcci.T)
    dist4i, cost4i, path4i = dtw(mfcc2.T, mfcci.T)
    dist5i, cost5i, path5i = dtw(mfcc3.T, mfcci.T)
    dist6i, cost6i, path6i = dtw(mfcc4.T, mfcci.T)
    dists[i] = (dist3i + dist4i + dist5i + dist6i)/4


plt.plot(dists)

word_match_idx = dists.argmin()
word_match_idx_bnds = np.array([word_match_idx,np.ceil(word_match_idx+window_size)])
samples_per_mfcc = 512
word_samp_bounds = (2/2) + (word_match_idx_bnds*samples_per_mfcc)

word = yX[word_samp_bounds[0]:word_samp_bounds[1]]

plt.plot(word)


IPython.display.Audio(data=word, rate=sr1)
IPython.display.Audio(data=y1, rate=sr1)
IPython.display.Audio(data=yX, rate=srX)
r = sr.Recognizer()
with sr.Microphone() as source:
	print("Say something!")
	audio = r.listen(source)
try:
	output = r.recognize_google(audio)
	o_word = output.split(' ')[0]
	if o_word.lower() in words:
		print("You said : " + o_word)
	else:
		print("Unknown")
except sr.UnknownValueError:
	print("Could not understand audio")
except sr.RequestError as e:
	print("Could not request results")

beg,end = word_match_idx_bnds[:]
beg,end = int(beg),int(end)
mfccWord = mfccX[:,beg:end]
librosa.display.specshow(mfccWord)

dist1X, cost1X, path1X = dtw(mfcc1.T, mfccWord.T)
dist1X, cost1X, path1X = dtw(mfcc1.T, mfccX.T)


