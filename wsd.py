
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from nltk import pos_tag            # import the pos_tag module
from math import log
from nltk.corpus import wordnet

def read_file(filename):
    infile1 = open(filename, "r")
    text1 = infile1.read()
    token_list1 = word_tokenize(text1.lower())      # Put individuals words into a list
    return token_list1

def clean_text(data):
    clean_data=[]
    stopw_list = stopwords.words("english")
    stopw_list.extend(("one","new","old","someone","go","used","wants","got",
                        "a", "an", "the", "The", "edit", "'s", "driver", "drivers", "may", "can",
                       "would", "usually", "also")) # Add words to the stopword list from nltk
    stop_words = set(stopw_list)
    for w in data:     # Only take non-stopwords and put them in a list
        if w not in stop_words and not any(ch.isdigit() for ch in w) and w.isalpha():#w.isdigit():
            clean_data.append(w)
    return clean_data

def freq_model(wordList):
    wordListProfile=nltk.FreqDist(wordList)
    frequent_words = wordListProfile.most_common(100000)
    return frequent_words

def hypernymWords(wordList):
    hypernymWords={}
    for i in wordList:
        syns = wordnet.synsets(i)
        hypernymWords[i]=syns.hypernym()
    return hypernymWords

def calculateProb(dataClass):
    model=[]
    total = 0
    for item in dataClass:
        total += item[1]
    freq = Counter(dict([ (x[0], float(x[1])/float(total))  for x in dataClass ]) )
    return freq

def pos(dataTokens):
    posTokens = [ (x[0], x[1][0]) for x in pos_tag(dataTokens) ]
    return posTokens

def buildModel(wordList,posTokens, probability, Class):
    reqformat=[]
    b=10000000   #### bias to make the
    for i in wordList:
        reqformat.append([i,posTokens[i],str(int(log(probability[i],2)*b)),Class])
    return reqformat

def findClass(tokens,fp1,fp2):
    resultPerson = 0.0
    resultTech= 0.0
    unknown=0.0000000001
    for token in tokens:
        resultTech += log(fp1.get(token,unknown ), 2)
        resultPerson += log(fp2.get(token,unknown),2)
    if max(resultTech, resultPerson) == resultTech: Class= "Technology"
    else:    Class= "Person"
    return Class

def formatFile(testfile,trainList,testList):
    trainfile = open('trainModel.txt', 'w')
    for item in trainList:
        trainfile.write("%s.\n" % ",".join(item))
    for x in testList:
        testfile.write("%s.\n" % ",".join(x))
    trainfile.close()
    testfile.close()



###########MAIN PROGRAM ###############

token_list1=read_file("Driver_technology.txt")
token_list2=read_file("Driving.txt")
test_tokensT=read_file("video_drivers.txt")
test_tokensP=read_file("truck_driver.txt")

print "Cleaning data...."
###### Feature 1: find pos
###### Feature 2: stop words Extraction

posTechnology= dict(pos(token_list1))
posPerson=dict(pos(token_list2))
posTestT= dict(pos(test_tokensT))
posTestP= dict(pos(test_tokensP))

filteredTechnology = []
filteredPerson = []

class1= clean_text(token_list1)
class2 = clean_text(token_list2)
testclassT=clean_text(test_tokensT)
testclassP=clean_text(test_tokensP)


####### Feature 3: Frequency and probability

freqClassTechnology= freq_model(class1)
freqClassPerson= freq_model(class2)
freqClassTestT= freq_model(testclassT)
freqClassTestP= freq_model(testclassP)
PTechnology= dict(calculateProb(freqClassTechnology))
PPerson=dict(calculateProb(freqClassPerson))
PTestT= dict(calculateProb(freqClassTestT))
PTestP= dict(calculateProb(freqClassTestP))

###### Feature 4: Removing common words

filteredTechnology=list(set(PTechnology.keys())-set(PPerson.keys()))
filteredPerson=list(set(PPerson.keys())-set(PTechnology.keys()))
for i in PTechnology.keys():
    if i not in filteredTechnology: del PTechnology[i]
for i in PPerson.keys():
    if i not in filteredPerson: del PPerson[i]

##### Train model in TiMBL format ### missing period at the end ######

print "Applying feature extraction...."
trainedTechnologyModel=buildModel(PTechnology.keys(),posTechnology,PTechnology,"Technology")
trainedPersonModel=buildModel(PPerson.keys(),posPerson,PPerson,"Person")
traineddata= trainedTechnologyModel+trainedPersonModel

######### Predict class #######

ClassT = findClass(testclassT,PTechnology,PPerson)
ClassP = findClass(testclassP,PTechnology,PPerson)

testClassModel=buildModel(PTestT.keys(),posTestT,PTestT,ClassT)
testfile = open('testModel_Technology.txt', 'w')
formatFile(testfile,traineddata,testClassModel)

testClassModel=buildModel(PTestP.keys(),posTestP,PTestP,ClassP)
testfile1=open('testModel_Driving','w')
formatFile(testfile1,traineddata,testClassModel)
print "Train and test model built."




