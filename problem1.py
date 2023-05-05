import random

from sklearn.datasets import load_iris
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
import pandas
import sys

names = ['variance', 'skew', 'curtosis', 'entropy', 'class']
banknotedata = pandas.read_csv('banknote_authentication.csv', names=names)
sample_num = len(banknotedata)


def splitRatio(trainpercent, banknoteData):
    trainSet = int(sample_num * trainpercent / 100)
    banknotedata = banknoteData.sample(frac=1) #to change the order of the data
    sample_train = np.array(banknotedata[['variance', 'skew', 'curtosis', 'entropy']].iloc[:trainSet])
    target_train = np.array(banknotedata['class'].iloc[:trainSet])
    sample_test = np.array(banknotedata[['variance', 'skew', 'curtosis', 'entropy']].iloc[trainSet:])
    target_test = np.array(banknotedata['class'].iloc[trainSet:])
    return sample_train, target_train, sample_test, target_test


def fixed_experiment(trainpercent,num):
    fixed = tree.DecisionTreeClassifier()

    for i in range(num):
        sample_train, target_train, sample_test, target_test = splitRatio(trainpercent, banknotedata)
        fixed = fixed.fit(sample_train, target_train)
        score = fixed.score(sample_test, target_test)
        print('Accuracy:' + str(score))
        print('size:' + str(fixed.tree_.node_count))
        # tree.plot_tree(fixed, filled=True, feature_names=names)
        # plt.show()


def random_experiment(trainpercent,num):
    random = tree.DecisionTreeClassifier()
    for i in range(5):
        meanscore = 0.0
        minscore=sys.float_info.max
        maxscore = sys.float_info.min
        meansize = 0.0
        minsize=sys.float_info.max
        maxsize=sys.float_info.min

        for j in range(num):
            sample_train, target_train, sample_test, target_test = splitRatio(trainpercent, banknotedata)
            random = random.fit(sample_train, target_train)
            score=random.score(sample_test, target_test) #calculating current score and size
            size=random.tree_.node_count
            if(score<minscore):
                minscore=score
            if(score>maxsize):
                maxscore=score
            if(size>maxsize):
                maxsize=size
            if(size<minsize):
                minsize=size

            meanscore += score
            meansize += size

            # tree.plot_tree(random, filled=True, feature_names=names)
            # plt.show()
        print(str(i+1)+':')
        print('train percent '+str(trainpercent)+'%')
        print('-----------------')
        print('min accuracy:'+str(minscore))
        print('max accuracy:'+str(maxscore))
        print('mean accuracy:' + str(meanscore/num))
        print('')
        print('min size:'+str(minsize))
        print('max size:'+str(maxsize))
        print('mean size:' + str(meansize/num))
        print('///////////////////////////////////////////////')
        trainpercent += 10
        meanAccuracies.append(meanscore/5)
        meanSizes.append(meansize/5)
    plt.plot(trainSetSize,meanAccuracies)
    plt.xlabel('train set size')
    plt.ylabel('mean accuracies')
    plt.show()
    plt.plot(trainSetSize,meanSizes)
    plt.xlabel('train set size')
    plt.ylabel('mean tree nodes')
    plt.show()

meanAccuracies=[]
meanSizes=[]
trainSetSize=[0.3,0.4,0.5,0.6,0.7]
exp_trials=5
print('fixed expirement:')
fixed_experiment(25,exp_trials)
print('random expirement:')
random_experiment(30,exp_trials)
# plt.plot(meanAccuracies,trainSetSize)
# plt.show()
# plt.plot(meanSizes,trainSetSize)
# plt.show()