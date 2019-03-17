import matplotlib
matplotlib.use('Agg')
import os
from itertools import *
import numpy as np
import torch
import cPickle
import copy
import lda
import random
import torch
import torch.nn as nn
import time
import sys
from torch.autograd import Variable
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from datetime import datetime
import argparse
import msgpack
import fire
import shutil
import logging
import torch.nn.init as init
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from subprocess import call
random.seed(0)
DATAMAPDIR = os.path.expanduser("ANALOGY/FB15k")
DATASETDIR = os.path.expanduser("ANALOGY/FB15k")



class BagOfWords(nn.Module):
    def __init__(self, wordCabSize, relationSize):
        super(BagOfWords, self).__init__()
        self.fc = nn.Linear(wordCabSize, relationSize)

    def forward(self, x):
        logit = self.fc(x)
        return logit

class CorNet(nn.Module):
    def __init__(self, relationSize):
        super(CorNet, self).__init__()
        hiddenDim = 200
        self.fc1 = nn.Linear(relationSize, hiddenDim)
        self.fc2 = nn.Linear(hiddenDim, relationSize)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x



class TBLogger():
    def __init__(self, logroot = 'output/runs', expName = None):
        self.logroot = logroot
        self.exptag = expName if expName is not None else datetime.now().strftime('%Y%m%d_%H%M%S')
        self.logpath = os.path.join(logroot, self.exptag)
        self.tbWriters = {}

    def getWritter(self, tbName):
        if not os.path.isdir(self.logpath):
            os.mkdir(self.logpath)
            #os.remove(os.path.join(self.logroot, "current_exp")) if os.path.lexists(os.path.join(self.logroot, "current_exp")) else None
            #os.system("ln -s %s %s" % (os.path.abspath(self.logpath), os.path.join(self.logroot, "current_exp")))
        if tbName not in self.tbWriters:
            tbWriter = SummaryWriter(os.path.join(self.logpath, tbName))
            self.tbWriters[tbName] = tbWriter
        else:
            tbWriter = self.tbWriters[tbName]
        return tbWriter

class PRPloter():
    def __init__(self, expName = 'NullExp'):
        self.fig, self.ax = plt.subplots()
        self.expName = expName

    def addCurve(self, precision, recall, fmt, label = 'Null'):
        self.ax.plot(recall, precision, fmt, label = label)

    def saveFig(self):
        self.ax.set_xlabel("recall")
        self.ax.set_ylabel("precision")
        self.ax.legend()
        plt.savefig('output/%s.pdf'%self.expName)
        plt.close(self.fig)

class MF(nn.Module):
    def __init__(self, entitySize, relationSize):
        super(MF, self).__init__()
        embDim = 5
        dtype = torch.FloatTensor
        self.headM = nn.Parameter(torch.randn(entitySize, embDim))
        self.relM = nn.Parameter(torch.randn(embDim, relationSize))
        self.bh = nn.Parameter(torch.randn(entitySize, 1))
        self.br = nn.Parameter(torch.randn(relationSize))

    def forward(self):
        pd = self.headM.mm(self.relM)
        return pd + self.bh.expand(pd.size()) + self.br.expand(pd.size())

class TransEModel(nn.Module):
    def __init__(self, entitySize, relationSize, embedDim = 100, gamma = 1.0):
        super(TransEModel, self).__init__()
        self.entityEmbed = nn.Embedding(entitySize, embedDim)
        init.xavier_uniform(self.entityEmbed.weight)
        self.relationEmbed = nn.Embedding(relationSize, embedDim)
        init.xavier_uniform(self.relationEmbed.weight)
        self.entitySize = entitySize
        self.gamma = gamma

    def calcdis(self, hrts):
        h = self.entityEmbed(hrts[:, 0])
        r = self.relationEmbed(hrts[:, 1])
        t = self.entityEmbed(hrts[:, 2])
        return torch.norm(h + r - t, p=1, dim=1, keepdim=True)

    def forward(self, hrts, neghrts):
        return torch.sum(F.relu(self.calcdis(hrts) - self.calcdis(neghrts) + self.gamma))

    def getTailDis(self, hrt):
        step = 200
        ranks = np.empty([len(hrt)]).astype('int64')
        for i in xrange(len(hrt) / step):
            begin = i * step
            end = (i + 1) * step
            dv = (self.entityEmbed(hrt[begin: end, 0]) + self.relationEmbed(hrt[begin: end, 1])).unsqueeze(1)
            dis = torch.norm(dv - self.entityEmbed.weight, p=2, dim=2, keepdim=False)
            _, rank = torch.sort(dis)
            ranks[begin: end] = (rank == hrt[begin: end, 2].unsqueeze(1)).nonzero().cpu().data.numpy()[:, 1]
        return ranks

    def calcAllTailScore(self, hrt):
        step = 200
        scores = []
        for i in xrange((len(hrt) + step - 1) / step):
            begin = i * step
            end = (i + 1) * step
            dv = (self.entityEmbed(hrt[begin: end, 0]) + self.relationEmbed(hrt[begin: end, 1])).unsqueeze(1)
            dis = torch.norm(dv - self.entityEmbed.weight, p=2, dim=2, keepdim=False)
            scores.append(dis.data.cpu())
        return torch.cat(scores)

    def getHeadDis(self, hrt):
        step = 200
        ranks = np.empty([len(hrt)]).astype('int64')
        for i in xrange(len(hrt) / step):
            begin = i * step
            end = (i + 1) * step
            dv = (self.entityEmbed(hrt[begin: end, 2]) - self.relationEmbed(hrt[begin: end, 1])).unsqueeze(1)
            dis = torch.norm(dv - self.entityEmbed.weight, p=2, dim=2, keepdim=False)
            _, rank = torch.sort(dis)
            ranks[begin: end] = (rank == hrt[begin: end, 0].unsqueeze(1)).nonzero().cpu().data.numpy()[:, 1]
        return ranks


class FactsDiscovery():
    def __init__(self, args = None, expName = None, cudaId = 0):
        if args is None:
            args = argparse.Namespace()
            args.inputTag = 'p0.02'
        self.expName = expName
        self.inputTag = '_' + str(args.inputTag)
        self.tbLogger = TBLogger(expName = self.expName)
        self.trs_vectorLen = 100
        self.prtH = 20
        self.prtR = 30
        self.prtR2 = 10# for making candindate set
        self.prtT = 2
        self.cudaId = cudaId
        self.setLogger()
        self.prPloter = PRPloter(self.expName)

    def setLogger(self):
        handlers = [logging.FileHandler('output/log/%s.log'%self.expName), logging.StreamHandler()]
        handlers[0].setLevel(logging.DEBUG)
        handlers[1].setLevel(logging.DEBUG)
        formatter = '%(asctime)s %(name)-12s %(message)s'
        handlers[0].setFormatter(formatter)
        handlers[1].setFormatter(formatter)
        self.log = logging.getLogger(self.expName)
        self.log.addHandler(handlers[0])
        self.log.addHandler(handlers[1])

    def saveStates(self, fileName):
        #state = [self.trs_W, self.cornetModel.state_dict(), self.cnnModel.state_dict(), self.cornetcnnModel.state_dict(), self.train_factsHistory]
        state = [self.cornetModel.state_dict(), self.cnnModel.state_dict(), self.cornetcnnModel.state_dict(), self.tbtModel.state_dict(), self.train_factsHistory]
        torch.save(state, open(fileName, 'wb'))
        #map(lambda x: self.cuda(x), [self.cornetModel, self.cnnModel, self.cornetcnnModel, self.tbtModel])

    def loadStates(self, fileName):
        cornetModelDict, cnnModelDict, cornetcnnModelDict, tbtModelDict, self.train_factsHistory = torch.load(open(fileName, 'rb'), map_location=lambda storage, loc: storage)
        self.cornetModel = CorNet(len(self.relation2id))
        self.cornetModel.load_state_dict(cornetModelDict)
        self.cnnModel = CNNRE(len(self.word2id), len(self.relation2id))
        self.cnnModel.load_state_dict(cnnModelDict)
        self.cornetcnnModel = CorNetCNN(len(self.word2id), len(self.relation2id))
        self.cornetcnnModel.load_state_dict(cornetcnnModelDict)
        self.tbtModel = TextBasedTransE(len(self.relation2id), len(self.entity2id), len(self.tbt_word2id), entityEmbdDim = 102, wordEmbdDim = 101)
        self.tbtModel.load_state_dict(tbtModelDict)
        map(lambda x: self.cuda(x), [self.cornetModel, self.cnnModel, self.cornetcnnModel, self.tbtModel])

    def cuda(self, var):
        if self.cudaId is not None:
            var = var.cuda(self.cudaId)
        return var

    def getTBWriter(self, tbName):
        return self.tbLogger.getWritter(tbName)

    def loadData(self, dataset = DATASETDIR, hrtOrder = 'htr'):
        print "data tag: ", self.inputTag
        entity2idFile = open(os.path.join(DATASETDIR, "entity2id.txt"))
        relation2idFile = open(os.path.join(DATASETDIR, "relation2id.txt"))
        word2idFile = open(os.path.join(DATAMAPDIR, "word2id.txt"))
        entityWordsFile = open(os.path.join(DATAMAPDIR, "entityWords.txt"))
        fb15kTrainFile = open(dataset + "-train.txt")
        fb15kTestFile = open(dataset + "-test.txt")
        fb15kValidFile = open(dataset + "-valid.txt")

        self.entity2id = {xp[0] : int(xp[1]) for xp in map(lambda x: x.split(), entity2idFile.readlines())}
        self.relation2id = {xp[0] : int(xp[1]) for xp in map(lambda x: x.split(), relation2idFile.readlines())}
        self.id2relation = [''] * len(self.relation2id)
        for i in self.relation2id:
            self.id2relation[self.relation2id[i]] = i
        self.id2entity = [''] * len(self.entity2id)
        for i in self.entity2id:
            self.id2entity[self.entity2id[i]] = i

        def getTripleFromFile(fileHandler):
            hrtIndex = map(lambda x: {'h': 0, 'r': 1, 't': 2}[x], hrtOrder)
            hrtIndex = np.argsort(hrtIndex)
            lines = imap(lambda x: x.split(), fileHandler.readlines())
            lines = imap(lambda x: [x[hrtIndex[0]], x[hrtIndex[1]], x[hrtIndex[2]]], lines)
            h_rt = {self.entity2id[k]: map(lambda x: [self.relation2id[x[1]], self.entity2id[x[2]]], v) for k, v in groupby(sorted(lines, key = lambda x: x[0]), key = lambda x: x[0])}
            facts = list(chain(*map(lambda h: map(lambda rt: (h, rt[0], rt[1]), h_rt[h]), h_rt.keys())))
            random.shuffle(facts)
            return h_rt, facts
        self.train_h_rt, self.train_facts = getTripleFromFile(fb15kTrainFile)
        self.train_factsHistory = []
        self.trs_WHistory = []
        self.test_h_rt, self.test_facts = getTripleFromFile(fb15kTestFile)
        self.test_sub_h_rt = {h: self.train_h_rt[h] if h in self.train_h_rt else [] for h in self.test_h_rt}
        self.valid_h_rt, self.valid_facts = getTripleFromFile(fb15kValidFile)
        self.valid_sub_h_rt = {h: self.train_h_rt[h] if h in self.train_h_rt else [] for h in self.valid_h_rt}
        def gethrFromhrt(hrt):
            return {k: list(set(map(lambda x: x[0], hrt[k]))) for k in hrt}
        self.train_h_r = gethrFromhrt(self.train_h_rt)
        self.test_h_r = gethrFromhrt(self.test_h_rt)
        self.test_sub_h_r = gethrFromhrt(self.test_sub_h_rt)
        self.valid_h_r = gethrFromhrt(self.valid_h_rt)
        self.valid_sub_h_r = gethrFromhrt(self.valid_sub_h_rt)
        self.train_keys = sorted(self.train_h_r.keys())
        self.test_keys = sorted(self.test_h_r.keys())
        self.valid_keys = sorted(self.valid_h_r.keys())

    # ======= Cor Net
    def CorNetmakeOneHot(self, featurehr, keys):
        feature = np.zeros((len(keys), len(self.relation2id)), dtype = 'float32')
        for i, h in enumerate(keys):
            if h not in featurehr: continue # means no head can be found in sub set.
            np.put(feature[i], featurehr[h], 1.0)
        ret = Variable(torch.from_numpy(feature), requires_grad = False)
        ret = self.cuda(ret)
        return ret

    def CorNetTrain(self):
        print "dataset size:", len(self.train_facts), len(self.test_facts), len(self.valid_facts)
        trainTarget = self.CorNetmakeOneHot(self.train_h_r, self.train_keys)
        testFeature = self.CorNetmakeOneHot(self.test_sub_h_r, self.test_keys)
        testTarget = self.CorNetmakeOneHot(self.test_h_r, self.test_keys)
        validFeature = self.CorNetmakeOneHot(self.valid_sub_h_r, self.valid_keys)
        validTarget = self.CorNetmakeOneHot(self.valid_h_r, self.valid_keys)

        epochNum = 1000
        self.cornetModel = CorNet(len(self.relation2id))
        self.cuda(self.cornetModel)
        optimizer = torch.optim.Adam(self.cornetModel.parameters(), lr = 0.005)#, weight_decay = 0.00001)
        criterion = nn.MultiLabelSoftMarginLoss()
        self.cornetModel.train()
        for epoch in range(epochNum):
            print self.expName + " Cornet epoch: %d"%epoch,
            trainFeature = np.zeros((len(self.train_keys), len(self.relation2id)), dtype = 'float32')
            for i, h in enumerate(self.train_keys):
                np.put(trainFeature[i], random.sample(self.train_h_r[h], int(max(1, random.random() * len(self.train_h_r[h])))), 1.0)
            trainFeature = Variable(torch.from_numpy(trainFeature), requires_grad = False)
            trainFeature = self.cuda(trainFeature)
            optimizer.zero_grad()
            logit = self.cornetModel(trainFeature)
            loss = criterion(logit, trainTarget)
            loss.backward()
            optimizer.step()
            print "train loss: %f "%loss.item(), '\033[F'


            # eval
            if epoch == epochNum - 1:
                self.CorNetEval(self.cornetModel, trainFeature, map(lambda x: self.train_h_r[x], self.train_keys), trainTarget, epoch, self.expName + 'cornet train' + self.inputTag)
                self.CorNetEval(self.cornetModel, validFeature, map(lambda x: self.valid_h_r[x], self.valid_keys), validTarget, epoch, self.expName + 'cornet valid' + self.inputTag)
                self.CorNetEval(self.cornetModel, testFeature, map(lambda x: self.test_h_r[x], self.test_keys), testTarget, epoch, self.expName + 'cornet test' + self.inputTag)
        print ''

    def CorNetEval(self, model, feature, target, targetOnehot, epoch, tbName):
        target = map(lambda x: np.array(x), target)
        tbWriter = self.getTBWriter(tbName)
        model.eval()
        criterion = nn.MultiLabelSoftMarginLoss()
        logit = model(feature)
        self.casestudy_cornetlogit = logit.data.cpu().numpy()
        loss = criterion(logit, targetOnehot)
        m, c = self.calcMAP(logit.data.cpu().numpy(), target)
        tbWriter.add_scalar('loss', loss.item(), epoch)
        tbWriter.add_scalar('MAP', m / c, epoch)
        #tbWriter.add_histogram('score', logit.item(), epoch)
        model.train()
        print tbName, "loss: ", loss.item()
        print tbName, "MAP: ", m / c

    # ======= predict tail entity
    def cornetPredictTail(self):
        self.pdtCalcRelationScoreMatrix()
        rank = np.empty_like(self.cornet_scoreMatrix, dtype='int32')
        sc = np.argsort(self.cornet_scoreMatrix)
        for i in range(rank.shape[0]):
            rank[i, sc[i]] = np.arange(rank.shape[1])[::-1]
        ranks, maps = self._evalRank(rank)
        self._precAndRecall(ranks)
        
        print "without tail relation filteration:"
        st = time.time()
        for self.prtR in [10]:#1345
            for self.prtT in [2]:
                print "self.prtR: %d, self.prtT: %d, self.prtH: %d"%(self.prtR, self.prtT, self.prtH)
                hrt, hrtscore = self.pdtCalcTail(self.cornet_scoreMatrix)
                self.pdtEvalTail(hrt, hrtscore)
        #self.pdtAddFactsToTrain()
        print "time: %.2f"%(time.time()- st)

    def pdtCalcRelationScoreMatrix(self):
        #heads, score = self.CorNetEval(self.cornetcnnModel, self.testWordsTargets, self.test_sub_h_r, 0, 'test')
        testFeature = self.CorNetmakeOneHot(self.test_sub_h_r, self.test_keys)
        score = self.cornetModel(testFeature).data.cpu().numpy()
        self.cornet_scoreMatrix = np.zeros([len(self.entity2id), len(self.relation2id)])
        self.cornet_scoreMatrix[self.test_keys] = score

    def pdtSaveRelationProb(self, path, prtR = 30):
        expscore = np.exp(self.cornet_scoreMatrix)
        prob = expscore / (1.0 + expscore)
        output = open(path, 'w')
        #prtR = 30#20#1345
        for h in self.test_keys:
            topn = np.argsort(prob[h])[ -prtR:]
            output.write('\n'.join(map(lambda x: self.id2entity[h] + '\t' + self.id2relation[x] + '\t' + str(prob[h, x]), topn)) + '\n')

        output.close()

    def pdtCalcTail(self, headRelationScore = None):
        h_topr_topt = {}
        hrt = np.empty(shape = (len(self.test_h_r) * self.prtH, 3), dtype = 'int32')
        hrt[:,:] = -1
        hrtscore = np.empty(shape = (len(self.test_h_r) * self.prtH))
        hrtscore[:] = -1
        hrtp = np.empty(shape = (self.prtR, 3), dtype = 'int64')
        hhrt = np.empty(shape = (self.prtR * self.prtT, 3), dtype = 'int32')
        hhrtscore = np.empty(shape = (self.prtR * self.prtT))
        #output = open('./ANALOGY/FB15k/%s-hr.txt'%self.inputTag[1:], 'w')
        for i, h in enumerate(self.test_h_r):
            print i, '/', len(self.test_h_r), '\033[F'
            topn = np.argsort(headRelationScore[h])[ -self.prtR:]
            hrtp[:, 0] = h
            hrtp[:, 1] = topn
            #output.write('\n'.join(map(lambda x: self.id2entity[h] + '\t' + self.id2relation[x] + '\t' + self.id2entity[0], topn)) + '\n')
            score = self.trsModel.calcAllTailScore(self.cuda(Variable(torch.from_numpy(hrtp), requires_grad = False)))
            _, rank = torch.sort(score)
            k = 0
            for bid, hrti in enumerate(hrtp):
                hhrt[k:k+self.prtT, 0] = hrti[0]
                hhrt[k:k+self.prtT, 1] = hrti[1]
                hhrt[k:k+self.prtT, 2] = rank[bid, :self.prtT]
                hhrtscore[k:k+self.prtT] = score[bid][rank[bid, :self.prtT]]
                k += self.prtT

            topn = np.argsort(hhrtscore)[:self.prtH]
            hrt[i * self.prtH : (i + 1) * self.prtH, :] = hhrt[topn, :]
            hrtscore[i * self.prtH : (i + 1) * self.prtH] = hhrtscore[topn]
            self.dbg_score = score
            self.dbg_hrt = hrt
            self.dbg_hrtscore = hrtscore
            self.dbg_hhrt = hhrt
            self.dbg_hhrtscore = hhrtscore
            self.dbg_topn = topn
        #output.close()
        self.pdt_hrt = hrt
        self.pdt_hrtscore = hrtscore
        return hrt, hrtscore

    def pdtEvalTail(self, hrts, hrtscore, addNew = False):
        self.pdt_ramainThrod = 100000# please change this if dataset changes
        tp = 0
        tt = 0
        topid = np.argsort(hrtscore)[:self.pdt_ramainThrod]
        #topid = range(len(hrtscore))
        maptmp = []
        validset = set(self.test_facts)
        for hrt in hrts[topid]:
            if tuple(hrt.tolist()) in validset:
                tp += 1
            tt += 1
            maptmp.append(1.*tp/tt)
        MAP, P, R = np.mean(maptmp), 1.0 * tp / tt, 1.0 * tp / len(self.test_facts)
        F1 = 2*P*R/(P+R)
        print 'map, precision, recall, F1, tp, tt'
        print '|%d|%d|%.4f|%.4f|%.4f|%.4f|%d|%d|'%(self.prtR, self.prtT, MAP, P, R, F1, tp, tt)

    def pdtMakeCandidateSet(self, prtR):
        r_h = {}
        for h in self.test_h_r:
            topn = np.argsort(self.cornetCNN_scoreMatrix[h])[ -prtR:].tolist()
            for r in topn:
                if r not in r_h:
                    r_h[r] = []
                r_h[r].append(h)
        return r_h

    def pdtCalcTailFiltered(self, mirror, headRelationScore = None):
        gpuStep = 200
        r_h = self.pdtMakeCandidateSet(self.prtR)
        mirror.pdtCalcRelationScoreMatrix()
        r_t = mirror.pdtMakeCandidateSet(self.prtR2)
        self.dbg_r_h = r_h
        self.dbg_r_t = r_t
        hrt = []
        print "Tail not found in relation: ", 
        for r in r_h:
            if r not in r_t:
                print r,
                continue
            hrt += [[x[0], r, x[1]] for x in product(r_h[r], r_t[r])]
        print ''
        hrt = self.cuda(Variable(torch.from_numpy(np.array(hrt)), requires_grad =False))
        ranks = np.empty([len(hrt)]).astype('int64')
        dis = []
        for i in xrange(len(hrt) / gpuStep):
            print i, '/', len(hrt) / gpuStep, '\033[F'
            begin = i * gpuStep
            end = (i + 1) * gpuStep
            dv = (self.trsModel.entityEmbed(hrt[begin: end, 0]) + self.trsModel.relationEmbed(hrt[begin: end, 1]) - self.trsModel.entityEmbed(hrt[begin: end, 2]))
            dis.append(torch.sum(torch.abs(dv), 1).data.cpu().numpy())
            torch.cuda.empty_cache()
        return hrt.data.cpu().numpy(), np.concatenate(dis)

    def pdtAddFactsToTrain(self):
        newFacts = []
        topid = np.argsort(self.pdt_hrtscore)[:self.pdt_ramainThrod]
        self.train_factsHistory.append(self.train_h_rt)
        for hrt in self.pdt_hrt[topid]:
            #newFacts.append(tuple(hrt.tolist()))
            if hrt[0] not in self.train_h_rt:
                self.train_h_rt[hrt[0]] = []
            self.train_h_rt[hrt[0]].append([hrt[1], hrt[2]])
        #self.trs_WHistory.append(self.trs_W)
        #self.train_facts = self.train_facts + newFacts
        #self.pdt_factsTrainNew = np.array(self.train_facts + newFacts, dtype = 'int32')

    def pdtReOptimize(self):
        #self.trs_W[:] = self.trs_baseW[:]
        gamma = 1.0
        batchNum = 100
        throd = 10000
        #factsTrain = self.factsTrain
        #===modify later
        first = True
        tmpr = self.pdt_hrt[np.argsort(self.pdt_hrtscore)[:throd]]
        if first:
            #tmpr[:, 1] = np.random.random_integers(0, len(self.relation2id) - 1, len(tmpr))
            #tmpr[:, 2] = np.random.random_integers(0, len(self.entity2id) - 1, len(tmpr))
            factsTrain = np.vstack([self.factsTrain, tmpr])
            self.pdt_newFactsTrain = factsTrain
        else:
            existSet = set(map(lambda x: tuple(x), self.pdt_newFactsTrain.tolist()))
            newid = []
            for i in range(len(tmpr)):
                if tuple(tmpr[i]) not in existSet:
                    newid.append(i)
            newf = tmpr[np.array(newid)]
            factsTrain = np.vstack([self.pdt_newFactsTrain, newf])
            self.pdt_newFactsTrain_1 = factsTrain
        #===modify later
        print factsTrain.shape
        updateids = self.test_h_r.keys()#assume we know the test head
        #validSet = self.trs_validSet
        validSet = self.trs_validSet.union(set(map(lambda x: tuple(x), self.pdt_hrt.tolist())))
        batchSize = factsTrain.shape[0] / batchNum
        epoch = 10
        decay_rate = 0.9999
        learning_rate = 0.001
        eps = 1e-8
        cache = np.zeros_like(self.trs_W)
        for ep in xrange(epoch):
            start = time.time()
            order = np.random.permutation(factsTrain.shape[0])
            for i in xrange(batchNum):
                sampleId = order[i * batchSize : (i + 1) * batchSize]
                negSample = factsTrain[sampleId]
                corids = np.random.choice([0, 2], batchSize)
                negSample[xrange(batchSize), corids] = np.random.random_integers(0, len(self.entity2id) - 1, batchSize)
                for j in xrange(len(negSample)):
                    while (negSample[j, 0], negSample[j, 1], negSample[j, 2]) in validSet:
                        corplc = np.random.choice([0, 2])
                        negSample[j, corplc] = random.randrange(0, self.trs_VE.shape[0])
                idLP = factsTrain[sampleId, 0]
                idRP = factsTrain[sampleId, 2]
                idLN = negSample[:, 0]
                idRN = negSample[:, 2]
                idRel = factsTrain[sampleId, 1]
                if ep % 100 == 0 and i%100 == 0:
                    print "epoch:", ep
                    print i,
                    print self.trsGetError(self.trs_W, idLP, idRP, idLN, idRN, idRel, gamma)
                dW = -learning_rate * self.trsGetGrad(self.trs_W, idLP, idRP, idLN, idRN, idRel, gamma)
                self.trs_W += dW
                #self.trs_VE[updateids] += -learning_rate * self.trsGetHeadGrad(self.trs_W, idLP, idRP, idLN, idRN, idRel, gamma)[updateids]
            nore = np.sqrt(np.sum(self.trs_VE**2, 1))
            noreg1 = np.where(nore > 1.0)
            self.trs_VE[noreg1] /= nore[noreg1].reshape((nore[noreg1].shape[0], 1))
            norl = np.sqrt(np.sum(self.trs_VL**2, 1))
            norlg1 = np.where(norl > 1.0)
            self.trs_VL[norlg1] /= norl[norlg1].reshape((norl[norlg1].shape[0], 1))
            finish = time.time()

    def _evalRank(self, methodRank, testSet = None):
        if testSet is None: testSet = self.test_h_r
        ranks = []
        maps = []
        for h in testSet:
            if h not in testSet: continue
            #rk = methodRank[self.test_G2L[h], np.array(testSet[h])]
            rk = methodRank[h, np.array(testSet[h])]
            ranks.append(rk)
            unique, counts = np.unique(rk, return_counts=True)
            maps.append(np.sum(np.arange(1, unique.shape[0] + 1) / (unique + 1.0) * counts) / np.sum(counts))
        return map(lambda x: np.sort(x), ranks), maps

    def _precAndRecall(self, ranks):
        print '|Threshold|Mean Precision|Mean Recall|\n|---|---|---|'
        for prt in [1, 5, 10, 20, 30, 40]:
            precision = []
            recall = []
            for rank in ranks:
                urank = np.unique(rank)
                gotnum = np.sum(urank < prt)
                recall.append(1.0 * gotnum / urank.shape[0])
                precision.append(1.0 * gotnum / prt)
            print '|', prt, ' ',
            print '|', '%.4f'%np.mean(precision),
            print '|', '%.4f'%np.mean(recall),
            print '|'


    # ====== MF
    def mfTrain(self, method = 'NNMF'):
        hiddenDim = 20
        rtMinFreq = 3
        rtsC = Counter(map(lambda x: (x[1], x[2]),self.train_facts))
        id2rt = list(set(filter(lambda x: rtsC[x] > rtMinFreq, rtsC)))
        self.mf_id2rt = id2rt
        rt2id = {rt:i for i, rt in enumerate(id2rt)}
        print len(id2rt), " pairs selected."
        X = np.zeros(shape = (len(self.entity2id), len(id2rt)))
        for h in self.train_h_rt:
            for rt in self.train_h_rt[h]:
                if tuple(rt) not in rt2id: continue
                X[h, rt2id[tuple(rt)]] = 1
        if method == 'NNMF':
            from sklearn.decomposition import NMF
            model = NMF(n_components=hiddenDim, init='random', random_state=0)
            W = model.fit_transform(X)
            H = model.components_
            self.mfScore = W.dot(H)
        elif method == 'LDA':
            from sklearn.decomposition import LatentDirichletAllocation
            lda = LatentDirichletAllocation(n_components=hiddenDim, max_iter=5,
                                            learning_method='online',
                                            learning_offset=50.,
                                            random_state=0)
            lda.fit(X)
            self.mfScore = lda.transform(X).dot(lda.components_)
        elif method == 'SVD':
            from sklearn.decomposition import TruncatedSVD
            svd = TruncatedSVD(n_components=hiddenDim, n_iter=7, random_state=42)
            svd.fit(X)
            self.mfScore = svd.transform(X).dot(svd.components_)

    def mfTest(self, method = ''):
        ranks = np.argsort(self.mfScore)
        prtRT = 50
        predicted = np.empty(shape = (len(self.test_h_rt) * prtRT, 3), dtype = 'int32')
        score = np.empty(len(self.test_h_rt) * prtRT)
        p = 0
        for h in self.test_h_rt:
            rts = np.array(map(lambda x: self.mf_id2rt[x], ranks[h, -prtRT:]))
            predicted[p : p + prtRT, 0] = h
            predicted[p : p + prtRT, 1] = rts[:, 0]
            predicted[p : p + prtRT, 2] = rts[:, 1]
            score[p : p + prtRT] = self.mfScore[h, ranks[h, -prtRT:]]
            p += prtRT
        self._evalTriples(predicted, score, method)
        # for P-R curve
        validset = set(self.test_facts)
        tfscore = zip(map(lambda x: int(tuple(x) in validset), predicted), score, map(lambda x: tuple(x), predicted))
        open('output/%s-%s-predict-score.txt'%(self.inputTag, method), 'w').write('\n'.join(map(lambda x: '\t'.join([str(x[0]), str(x[1]), self.id2entity[x[2][0]], self.id2relation[x[2][1]], self.id2entity[x[2][2]]]), tfscore)))
        #self.prPloter.addCurve(P, R, 'r--', method)

    def _evalTriples(self, predicted, score, method = ''):
        if type(predicted) is np.ndarray:
            predicted = predicted.tolist()
        predicted = map(lambda x: tuple(x), predicted)
        filterset = set(self.train_facts).union(set(self.valid_facts))
        validset = set(self.test_facts)
        tp = tt = 0
        maptmp = []
        #y_true = np.array(map(lambda x: int(x in validset), predicted), dtype = 'int32')
        #P, R, T = precision_recall_curve(y_true, score)
        #return P, R
        for i, hrt in enumerate(predicted):
            thrt = tuple(hrt)
            #if thrt in filterset and thrt not in validset: continue
            tt += 1
            if tuple(hrt) in validset:
                tp += 1
                maptmp.append(1.*tp/tt)
            #if tt == 20000: break
        MAP, P, R = np.mean(maptmp), 1.0 * tp / tt, 1.0 * tp / len(self.test_facts)
        F1 = 2*P*R/(P+R)
        print 'method, map, precision, recall, F1, tp, tt'
        print '|%s|%.4f|%.4f|%.4f|%.4f|%d|%d|'%(method, MAP, P, R, F1, tp, tt)

    # ======= cor
    def corCalc(self, h_r = None):
        if h_r is None: h_r = self.train_h_r
        relationEntity = np.zeros((len(self.relation2id), len(self.entity2id)))
        for h in h_r:
            for r in h_r[h]:
                relationEntity[r, h] = 1
        #for h in self.test_sub_h_r: # use test seed triples to frac
        #    for r in self.test_sub_h_r[h]:
        #        relationEntity[r, h] = 1
        relationCorr = np.corrcoef(relationEntity)
        ids = np.isnan(relationCorr)
        relationCorr[ids] = 0.0
        self.cor_relationRelationCorr = relationCorr
        self.corTest(self.train_h_r, self.train_h_r, self.train_keys, "cor train")
        self.corTest(self.valid_sub_h_r, self.valid_h_r, self.valid_keys, "cor valid")
        self.corTest(self.test_sub_h_r, self.test_h_r, self.test_keys, "cor test")

    def corTest(self, hr_sub, hr, keys, tbName, entityRelationCount = None, relationRelationCorr = None):
        entityRelation = np.zeros((len(self.entity2id), len(self.relation2id)))
        for h in hr_sub:
            for r in hr_sub[h]:
                entityRelation[h, r] = 1
        testCorr = entityRelation.dot(self.cor_relationRelationCorr)
        tmap, tcnt = self.calcMAP(testCorr[np.array(keys)], map(lambda x: hr[x], keys))
        print tbName,"MAP: ", tmap / tcnt

    #=======Direct SVD
    def dsvdCalc(self):
        entityRelation = np.zeros((len(self.entity2id), len(self.relation2id)))
        for h in self.train_h_r:
            for r in self.train_h_r[h]:
                entityRelation[h, r] += 1
        for h in self.valid_sub_h_r: # use test seed triples to frac
            for r in self.valid_sub_h_r[h]:
                entityRelation[h, r] += 1
        for h in self.test_sub_h_r: # use test seed triples to frac
            for r in self.test_sub_h_r[h]:
                entityRelation[h, r] += 1
        u,s,v = np.linalg.svd(entityRelation)
        self.dsvd_entityRelationCount = entityRelation
        self.dsvd_U = u
        self.dsvd_S = s
        self.dsvd_V = v
        self.dsvdTest(self.train_h_r, self.train_keys, "svd train")
        self.dsvdTest(self.valid_h_r, self.valid_keys, "svd valid")
        self.dsvdTest(self.test_h_r, self.test_keys, "svd test")

    def dsvdTest(self, hr, keys, tbName, dsvdLen = None):
        U, S, V = self.dsvd_U, self.dsvd_S, self.dsvd_V
        if dsvdLen is None:
            dsvdLen = 5
        s = copy.copy(S)
        s[dsvdLen:] = 0.0
        minl = min(U.shape[1], V.shape[0])
        u = U[:, :minl] * np.sqrt(s)
        v = V[:minl, :] * np.sqrt(s).reshape((s.shape[0], 1))
        score = u.dot(v)
        tmap, tcnt = self.calcMAP(score[np.array(keys)], map(lambda x: hr[x], keys))
        print tbName, " MAP: ", tmap / tcnt

    # ======= MF
    def mfCalc(self):
        trainTargetList = map(lambda x: self.train_h_r[x], self.train_keys) + map(lambda x: self.valid_sub_h_r[x], self.valid_keys) + map(lambda x: self.test_sub_h_r[x], self.test_keys)
        validTargetList = map(lambda x: self.valid_h_r[x], self.valid_keys)
        testTargetList = map(lambda x: self.test_h_r[x], self.test_keys)
        entityRelation = np.zeros((len(self.entity2id), len(self.relation2id)), dtype = 'float32')
        for h in self.train_h_r:
            for r in self.train_h_r[h]:
                entityRelation[h, r] = 1
        for h in self.test_sub_h_r: # use test seed triples to frac
            for r in self.test_sub_h_r[h]:
                entityRelation[h, r] = 1
        for h in self.valid_sub_h_r: # use test seed triples to frac
            for r in self.valid_sub_h_r[h]:
                entityRelation[h, r] = 1
        testEntityRelation = np.zeros((len(self.entity2id), len(self.relation2id)), dtype = 'float32')
        for h in self.test_keys:
            for r in self.test_h_r[h]:
                testEntityRelation[h, r] = 1
        validEntityRelation = np.zeros((len(self.entity2id), len(self.relation2id)), dtype = 'float32')
        for h in self.valid_keys:
            for r in self.valid_h_r[h]:
                validEntityRelation[h, r] = 1
        entityRelation = Variable(torch.from_numpy(entityRelation), requires_grad = False)
        validEntityRelation = Variable(torch.from_numpy(validEntityRelation), requires_grad = False)
        testEntityRelation = Variable(torch.from_numpy(testEntityRelation), requires_grad = False)
        if cudaId is not None: entityRelation, validEntityRelation, testEntityRelation = entityRelation.cuda(cudaId), validEntityRelation.cuda(cudaId), testEntityRelation.cuda(cudaId)
        epochs = 1000
        self.mfModel = MF(len(self.entity2id), len(self.relation2id))
        if cudaId is not None: self.mfModel.cuda(cudaId)
        optimizer = torch.optim.Adam(self.mfModel.parameters(), lr = 0.1)#, weight_decay = 0.00001)
        #criterion = nn.MSELoss()
        criterion = nn.MultiLabelSoftMarginLoss()
        for epoch in range(epochs):
            print self.getExpTag(), "MF epoch: ", epoch
            optimizer.zero_grad()
            logit = self.mfModel()
            loss = criterion(logit, entityRelation)
            print "train loss: ", loss.item()
            loss.backward()
            optimizer.step()
            self.mfEval(self.mfModel, entityRelation, trainTargetList, np.array(self.train_keys + self.valid_keys + self.test_keys), epoch, "train")
            self.mfEval(self.mfModel, validEntityRelation, validTargetList, np.array(self.valid_keys), epoch, "valid")
            self.mfEval(self.mfModel, testEntityRelation, testTargetList, np.array(self.test_keys), epoch, "test")

    def mfEval(self, model, target, targetList, index, epoch, tbName):
        tbWriter = self.getTBWriter(tbName)
        if cudaId is not None: indexp = torch.from_numpy(index.astype('int64')).cuda(cudaId)
        #criterion = nn.MSELoss()
        criterion = nn.MultiLabelSoftMarginLoss()
        model.eval()
        logit = self.mfModel()
        loss = criterion(logit[indexp], target[indexp])
        m, c = self.calcMAP(logit[indexp].data.cpu().numpy(), targetList)
        tbWriter.add_scalar('loss', loss.item(), epoch)
        tbWriter.add_scalar('MAP', m / c, epoch)
        model.train()
        print tbName, "loss: ", loss.item()
        print tbName, "MAP: ", m / c


    # ======= topic
    def conCalc(self):
        entityRelation = np.zeros((len(self.entity2id), len(self.relation2id)), dtype = 'int32')
        for h in self.train_h_r:
            for r in self.train_h_r[h]:
                entityRelation[h, r] = 1
        for h in self.test_sub_h_r: # use test seed triples to frac
            for r in self.test_sub_h_r[h]:
                entityRelation[h, r] = 1
        for h in self.valid_sub_h_r: # use test seed triples to frac
            for r in self.valid_sub_h_r[h]:
                entityRelation[h, r] = 1
        topicNum = 5
        model = lda.LDA(n_topics=topicNum, n_iter=200, random_state=1)
        model.fit(entityRelation)
        #relation_concept = model.topic_word_.T
        #relation_concept /= np.sum(relation_concept, 1).reshape(relation_concept.shape[0], 1)
        self.con_entityRelation = entityRelation
        self.con_entityConcept = model.doc_topic_
        self.con_conceptRelation = model.topic_word_
        self.con_relationConcept = model.topic_word_.T

    def conTest(self):
        entityConcept = self.con_entityConcept#[np.array(self.test_L2G)]
        conceptRelation = self.con_conceptRelation
        entityRelation = entityConcept.dot(conceptRelation)
        tmap, tcnt = self.calcMAP(entityRelation[np.array(self.train_keys)], map(lambda x: self.train_h_r[x], self.train_keys))
        print "topic train MAP: ", tmap / tcnt
        tmap, tcnt = self.calcMAP(entityRelation[np.array(self.valid_keys)], map(lambda x: self.valid_h_r[x], self.valid_keys))
        print "topic valid MAP: ", tmap / tcnt
        tmap, tcnt = self.calcMAP(entityRelation[np.array(self.test_keys)], map(lambda x: self.test_h_r[x], self.test_keys))
        print "topic test MAP: ", tmap / tcnt

    # ======= TransE
    def trsTrain(self):
        import time
        tbWriter = SummaryWriter("runs/TransEBaseline")
        epochN = 1000
        batchSize = 100000
        #batchSize = len(self.train_facts) / 100
        self.trsModel = TransEModel(len(self.entity2id), len(self.relation2id), self.trs_vectorLen)
        self.cuda(self.trsModel)
        optimizer = torch.optim.SGD(self.trsModel.parameters(), lr = 0.001)
        factsTrain = self.cuda(Variable(torch.from_numpy(np.array(self.train_facts)), requires_grad = False))

        for epoch in range(epochN):
            print self.expName, " TransE epoch:%d "%epoch, 
            start = time.time()
            batch = 0
            negFactsTrain = np.array(self.train_facts)
            nid = np.random.choice([0, 2], len(negFactsTrain))
            net = np.random.random_integers(0, len(self.entity2id) - 1, len(negFactsTrain)).astype('int32')
            negFactsTrain[np.arange(len(negFactsTrain)), nid] = net
            negFactsTrain = self.cuda(Variable(torch.from_numpy(negFactsTrain), requires_grad = False))
            tloss = 0.
            self.trsModel.entityEmbed.weight.data /= self.trsModel.entityEmbed.weight.data.norm(p=2, dim=1, keepdim=True)
            self.trsModel.relationEmbed.weight.data /= self.trsModel.relationEmbed.weight.data.norm(p=2, dim=1, keepdim=True)
            #self.trsModel.entityEmbed.weight.data[:, :] = F.normalize(self.trsModel.entityEmbed.weight.data, p = 1, dim=1)[:, :]
            #self.trsModel.relationEmbed.weight.data[:, :] = F.normalize(self.trsModel.relationEmbed.weight.data, p = 1, dim=1)[:, :]
            while batch * batchSize < len(self.train_facts):
                optimizer.zero_grad()
                loss = self.trsModel(factsTrain[batch * batchSize : (batch + 1) * batchSize], negFactsTrain[batch * batchSize : (batch + 1) * batchSize])
                #print loss.item()
                tloss += loss.item()
                loss.backward()
                optimizer.step()
                batch += 1
            print "loss: %f\t"%(tloss / len(self.train_facts)), '\033[F'
            tbWriter.add_scalar('loss', tloss / len(self.train_facts), epoch)
        print ''

    def trsTest(self):
        factsTrain = self.cuda(Variable(torch.from_numpy(np.array(self.train_facts)), requires_grad = False))
        factsTest = self.cuda(Variable(torch.from_numpy(np.array(self.test_facts)), requires_grad = False))
        tailRank = self.trsModel.getTailDis(factsTest)
        headRank = self.trsModel.getHeadDis(factsTest)
        self.testTailRank, self.testHeadRank = tailRank, headRank
        tc = np.sum((tailRank < 10).astype('int32'))
        hc = np.sum((headRank < 10).astype('int32'))
        print "predict tail: ", tc * 1.0 / len(tailRank), "\tpredict head: ", hc * 1.0 / len(headRank), "mean: ", (hc + tc) / 2.0 / len(tailRank)

    # ======= call Analogy program
    def algPredictTail(self):
        pass


    # ======= common utils
    def calcMAP(self, scoreMatrix, trueRelations):
            sc = scoreMatrix.argsort()[:, ::-1]
            maps = map(lambda x: np.arange(1, len(x) + 1) * 1. / x, imap(lambda x: (np.where(np.in1d(x[0], x[1]))[0] + 1.), izip(sc, trueRelations)))
            totalMap = sum(map(lambda x: x.sum(), maps))
            totalCnt = sum(map(lambda x: len(x), maps))
            return totalMap, totalCnt


    #==== run codes
    def mergeFile(self, f1, f2, f3):
        lines1 = set(open(f1).readlines())
        lines2 = set(open(f2).readlines())
        lines3 = list(lines1.union(lines2))
        open(f3, 'w').write(''.join(lines3))

    def runTrainCorNet(self):
        self.CorNetTrain()

    def runTransE(self):
        self.trsTrain()

    def run(self, step, expName = 'NullExp', cudaId = None, inputTag = ''):
        self.tailIndexedModel = FactsDiscovery()
        self.args = self.tailIndexedModel.args = argparse.Namespace()
        self.expName = expName
        self.tailIndexedModel.expName = expName+'_tailIndexed'
        self.cudaId = self.tailIndexedModel.cudaId = cudaId
        self.inputTag = self.tailIndexedModel.inputTag = inputTag

        #self.loadData()
        if step == 'trainCorNet':
            self.loadData('./ANALOGY/FB15k/%s'%self.inputTag, 'hrt')
            self.runTrainCorNet()
            self.pdtCalcRelationScoreMatrix()
            self.pdtSaveRelationProb('./ANALOGY/FB15k/%s-hr.txt'%self.inputTag)

            self.tailIndexedModel.loadData('./ANALOGY/FB15k/%s'%self.inputTag, 'trh')
            self.tailIndexedModel.runTrainCorNet()
            self.tailIndexedModel.pdtCalcRelationScoreMatrix()
            self.tailIndexedModel.pdtSaveRelationProb('./ANALOGY/FB15k/%s-rt.txt'%self.inputTag)

        elif step == 'baseline':
            self.loadData('./ANALOGY/FB15k/%s'%self.inputTag, 'hrt')
            self.runTrainCorNet()
            self.mfTrain('NNMF') or self.mfTest('NNMF')
            self.mfTrain('SVD') or self.mfTest('SVD')
            self.mfTrain('LDA') or self.mfTest('LDA')
        elif step == 'feedback':
            os.system("cp ./ANALOGY/FB15k/%s-train.txt ./output/%s-train.txt"%(self.inputTag, self.inputTag))
            os.system("cp ./ANALOGY/FB15k/%s-train.txt ./ANALOGY/FB15k/%s-iter-train.txt"%(self.inputTag, self.inputTag))
            os.system("cp ./ANALOGY/FB15k/%s-test.txt ./output/%s-test.txt"%(self.inputTag, self.inputTag))
            os.system("cp ./ANALOGY/FB15k/%s-valid.txt ./output/%s-valid.txt"%(self.inputTag, self.inputTag))
            for i in range(3):
                print ('====== iter %d ====='%i)
                self.loadData('./output/%s'%self.inputTag, 'hrt')
                self.runTrainCorNet()
                self.pdtCalcRelationScoreMatrix()
                self.pdtSaveRelationProb('./ANALOGY/FB15k/%s-hr.txt'%self.inputTag)
                os.system('./ANALOGY/main -algorithm Analogy -model_path ./output/Analogy_FB15k_p0.5.model -dataset ./ANALOGY/FB15k/p0.5 -prediction true')
                self.mergeFile('./ANALOGY/FB15k/%s-iter-train.txt'%self.inputTag, './output/%s-train.txt'%self.inputTag, './output/%s-train.txt'%self.inputTag)

def mkdir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)



if __name__ == "__main__":
    from facts_discovery import *
    mkdir('output')
    mkdir('output/log')
    mkdir('output/runs')
    self = FactsDiscovery(cudaId = None)
    s = self
    fire.Fire(s)
