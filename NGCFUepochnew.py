import torch
from torch import nn as nn
from toyDataset.loaddata import *
from scipy.sparse import coo_matrix
import pandas as pd
import numpy as np
from numpy import diag 
from torch.utils.data import DataLoader 
from torch.utils.data import random_split
from torch.optim import Adam
from torch.nn import MSELoss
from GraphNCF.dataPreprosessing import ML1K
from torch.nn import Module 
from scipy.sparse import vstack
from scipy import sparse 

from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine

from parse import parse_args 
from torch.nn import init
 
global Uijdict ,Uijchange,hnnz, Iijdict ,Iijchange,LaplacianMat,UILaplacianMat,utopk,itopk,Ruu,Rii
args = parse_args()
SEED = args.seed #np.random.randint(10000)  #
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

global dk,rtA,mode,um,im 
mode=args.mode
dk=args.dk
para = {
    'epoch':60,
    'lr':0.01,
    'batch_size':2048,
    'train':0.9
}

trainrt,testrt,DATASET=loaddt(args.dataset)
layer_size=32
embed_size=32
    
userNum =max(trainrt['userId'].max(),testrt['userId'].max())+1
itemNum =max(trainrt['itemId'].max(),testrt['itemId'].max())+1   

  
train = ML1K(trainrt)
test = ML1K(testrt)
rt=trainrt
  
print(DATASET,userNum,itemNum)
 

para['epoch']=args.epoch
para['lr']=args.lr
para['batch_size']=args.batch_size
lysz=args.lysz


utopk=args.utopk
itopk=args.itopk 

if args.UItype==1:
    Ruu=True
    Rii=False
    methord="embepochUww "
if args.UItype==2:
    Ruu=False
    Rii=True
    methord="embepochIww "
if args.UItype==3:
    Ruu=True
    Rii=True
    methord="embepochUIww "
 


Uijdict= { }
Uijchange={}
Iijdict= { }
Iijchange={}
for ui in range(userNum):
    Uijdict[ui]={} 
for ui in range(itemNum):
    Iijdict[ui]={}

  
# LaplacianMat=sparse.identity(userNum+itemNum).todok()
# UILaplacianMat=sparse.identity(userNum+itemNum).todok()

class GNNLayer(Module):

    def __init__(self,inF,outF):

        super(GNNLayer,self).__init__()
        self.inF = inF
        self.outF = outF
        self.linear = torch.nn.Linear(in_features=inF,out_features=outF)
        self.linear1 = torch.nn.Linear(in_features=inF,out_features=outF)
        self.interActTransform = torch.nn.Linear(in_features=inF,out_features=outF)
        self.interActTransform1 = torch.nn.Linear(in_features=inF,out_features=outF)
        self.coefficient1 = torch.nn.Parameter(torch.Tensor([1.0]))
        self.coefficient2 = torch.nn.Parameter(torch.Tensor([1.0]))
        self.coefficient3 = torch.nn.Parameter(torch.Tensor([1.0]))
        self.coefficient4 = torch.nn.Parameter(torch.Tensor([1.0]))


    def forward(self, laplacianMat,selfLoop,UILaplacianMat,features):
        # for GCF ajdMat is a (N+M) by (N+M) mat
        # laplacianMat L = D^-1(A)D^-1 # 拉普拉斯矩阵
       
        L1 = laplacianMat #+ selfLoop
        L2 = laplacianMat.cuda()
        L1 = L1.cuda()
        L3=UILaplacianMat.cuda()

        inter_feature = torch.mul(features,features)
        inter_part1 = self.linear(torch.sparse.mm(L1,features))
      #  inter_partself=self.linear(torch.sparse.mm(selfLoop.cuda(),features))
        inter_part2 = self.interActTransform(torch.sparse.mm(L2,inter_feature))  
       
        inter_partw = self.linear1(torch.sparse.mm(L3,features))
        inter_partw2 = self.interActTransform1(torch.sparse.mm(L3,inter_feature))  

 
        #return  self.coefficient1*inter_part1+self.coefficient2*inter_part2+self.coefficient3*inter_partw +self.coefficient4*inter_partw2    #self.interActTransform(intcat) # self.coefficient*inter_part1+self.coefficient1*inter_part2 #
        return  inter_part1+inter_partw +inter_part2+inter_partw2    #self.interActTransform(intcat) # self.coefficient*inter_part1+self.coefficient1*inter_part2 #
       # return  torch.cat([ inter_part1,inter_part2 ],1) 
class GCF(Module):

    def __init__(self,userNum,itemNum,rt,embedSize=64,layers=[64,64],useCuda=True):

        super(GCF,self).__init__()
        global LaplacianMat,UILaplacianMat,Ruu,Rii,um,im

        self.useCuda = useCuda
        self.userNum = userNum
        self.itemNum = itemNum
        self.uEmbd = nn.Embedding(userNum,embedSize)
        self.iEmbd = nn.Embedding(itemNum,embedSize)
        self.GNNlayers = torch.nn.ModuleList()
        # LaplacianMat = self.buildLaplacianMat(rt) # sparse format
        # UILaplacianMat=self.builddyUI(rt)
        LaplacianMat,UILaplacianMat=self.syn_buildLaplacianMat(rt,1)
 
  
        self.leakyRelu = nn.LeakyReLU()
        self.selfLoop = self.getSparseEye(self.userNum+self.itemNum)

        self.transForm1 = nn.Linear(in_features=layers[-1]*(len(layers))*2,out_features=64)
        self.transForm2 = nn.Linear(in_features=64,out_features=32)
        self.transForm3 = nn.Linear(in_features=32,out_features=1)
        


        self.transFormcat = nn.Linear(in_features=layers[-1]*(len(layers))*3,out_features=64)

        for From,To in zip(layers[:-1],layers[1:]):
            self.GNNlayers.append(GNNLayer(From,To))

    def getSparseEye(self,num):
        i = torch.LongTensor([[k for k in range(0,num)],[j for j in range(0,num)]])
        val = torch.FloatTensor([1]*num)
        return torch.sparse.FloatTensor(i,val)

    def syn_buildLaplacianMat(self,rt,finalEmbd,uuflg=False,iiflg=False):
        global LaplacianMat ,UILaplacianMat,hnnz,utopk,itopk,Uijdict

        rt_item = rt['itemId'] + self.userNum
        
        # if rtA.find("1")>-1:
        #     uiMat = coo_matrix(([1]*len(rt['rating']), (rt['userId'], rt['itemId'])))  
        #     uim=coo_matrix(([1]*len(rt['rating']), (rt['userId'], rt['itemId']))) 
        #     uim2=coo_matrix(([1]*len(rt['rating']), (rt['userId'], rt['itemId']))) 
        #     uiMat_upperPart = coo_matrix(([1]*len(rt['rating']), (rt['userId'], rt_item)))
        # else:
        
        uiMat = coo_matrix((rt['rating'], (rt['userId'], rt['itemId'])))
        uim=coo_matrix((rt['rating'], (rt['userId'], rt['itemId'])))
        uim2=coo_matrix((rt['rating'], (rt['userId'], rt['itemId'])))
        uiMat_upperPart = coo_matrix((rt['rating'], (rt['userId'], rt_item)))

        uiMat = uiMat.transpose()
        uiMat.resize((self.itemNum, self.userNum + self.itemNum))
        uiMat_upperPart.resize((self.userNum, self.userNum + self.itemNum))

        A0 = sparse.vstack([uiMat_upperPart,uiMat])
        uutmp=sparse.dok_matrix((self.userNum,self.userNum + self.itemNum), dtype=np.float32) 
        
        um=0
        im=0

        if uuflg:

            uidx = torch.LongTensor([i for i in range(self.userNum)]) 
            if self.useCuda == True:
                uidx = uidx.cuda()
            uim = finalEmbd[uidx].squeeze().cpu().detach().numpy()

  
            uu = 1-pairwise_distances(uim, metric="cosine")#cosine
            uutopk=np.argsort(uu, axis=1)
            uutopk=uutopk[:,-utopk-1:] 
             
            for  i in range(self.userNum):
                for tk in uutopk[i]:
                    uutmp[i,tk]=uu[i,tk]  
             
        iitmp=sparse.dok_matrix((self.itemNum,self.userNum + self.itemNum), dtype=np.float32) 
        if iiflg:
 
            iidx = torch.LongTensor([i for i in range(self.itemNum)])+ self.userNum
            if self.useCuda == True: 
                iidx = iidx.cuda()
            uim = finalEmbd[iidx].squeeze().cpu().detach().numpy()

            uu = 1-pairwise_distances(uim, metric="cosine")#cosine
            # print("i",uu)

            uutopk=np.argsort(uu, axis=1)
            uutopk=uutopk[:,-itopk-1:]
            
             
            for i in range(self.itemNum): 
                for tp in uutopk[i]: 
                    iitmp[i,tp]=uu[i,tp] 

#        print("uim--------------------------------------",np.array(list(uutmp.values())).mean(),np.array(list(iitmp.values())).mean())     
        Aui = sparse.vstack([uutmp,iitmp])

        selfLoop = sparse.eye(self.userNum+self.itemNum)
           
        A0=A0+selfLoop#*0.0001
 
        # if rtA.find("0")>-1:
        #     sumArr =  (A0>0).sum(axis=1)#
        #     sumArr2 =  (A0>0).sum(axis=0)# 
        # else:#(1,rtA)
        sumArr =  A0.sum(axis=1)#
        sumArr2 =  A0.sum(axis=0)# 

        diag = list(np.array(sumArr.flatten())[0] )
        diag = np.power(diag,-0.5)
        D = sparse.diags(diag)
        L = D * A0 * D 
        L = sparse.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row,col])
        data = torch.FloatTensor(L.data)
        SparseA0 = torch.sparse.FloatTensor(i,data)

       
        Aui=Aui+selfLoop #

        # if rtA.find("0")>-1:
        #     sumArr =  (Aui>0).sum(axis=1)#
        #     sumArr2 =  (Aui>0).sum(axis=0)# 
        # else:#(1,rtA)
        
        sumArr =  Aui.sum(axis=1)#
        sumArr2 = Aui.sum(axis=0)#  


        # diag = list(np.array(sumArr.flatten())[0] )
        # diag = np.power(diag,-0.5)
        # D = sparse.diags(diag)
 
        # diag = list( np.array(sumArr2.flatten())[0])
        # diag = np.power(diag,-0.5)
        # D2 = sparse.diags(diag)
        # L = D * Aui * D2 
       
        diag = list( 1./np.array(sumArr.flatten())[0]) 
        D = sparse.diags(diag)
        L = D * Aui


        L = sparse.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row,col])
        data = torch.FloatTensor(L.data)
        SparseAui = torch.sparse.FloatTensor(i,data)
 
        return SparseA0,SparseAui

 
    def getFeatureMat(self):
        uidx = torch.LongTensor([i for i in range(self.userNum)])
        iidx = torch.LongTensor([i for i in range(self.itemNum)])
        if self.useCuda == True:
            uidx = uidx.cuda()
            iidx = iidx.cuda()

        userEmbd = self.uEmbd(uidx)
        itemEmbd = self.iEmbd(iidx)
        features = torch.cat([userEmbd,itemEmbd],dim=0)
        return features

    def forward(self,userIdx,itemIdx):

        itemIdx = itemIdx + self.userNum
        userIdx = list(userIdx.cpu().data)
        itemIdx = list(itemIdx.cpu().data)
        # gcf data propagation
        features = self.getFeatureMat()
        finalEmbd = features.clone()
        for gnn in self.GNNlayers:
            features = gnn(LaplacianMat,self.selfLoop ,UILaplacianMat,features)
            #features = gnn(LaplacianMat+UILaplacianMat,self.selfLoop ,features)
            features = nn.ReLU()(features)
            finalEmbd = torch.cat([finalEmbd,features.clone()],dim=1)

        userEmbd = finalEmbd[userIdx]
        itemEmbd = finalEmbd[itemIdx]
        embd = torch.cat([userEmbd,itemEmbd],dim=1)

        embd = nn.ReLU()(self.transForm1(embd)) 
        embd = self.transForm2(embd)
        embd = self.transForm3(embd)
        prediction = embd.flatten()
        # uu =torch.mm(userEmbd,  userEmbd.transpose(0,1)).cpu().detach().numpy()
        # print(np.max(uu) )

        return prediction,userEmbd,itemEmbd,finalEmbd
  
 
dl = DataLoader(train,batch_size=para['batch_size'],shuffle=True,pin_memory=True)

lysz=args.lysz
embed_size=layer_size=args.embed_size 

model = GCF(userNum, itemNum, rt, embed_size,layers=[layer_size,]*lysz).cuda()
optim = Adam(model.parameters(), lr=para['lr'],weight_decay=args.decay)
lossfn = MSELoss()
testdl = DataLoader(test,batch_size=len(test),)


def initNetParams(net):
    '''Init net parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Linear):
            init.kaiming_uniform_(m.weight)
            init.constant_(m.bias, 0)
        elif isinstance(m,nn.Embedding):
            m.weight.data.normal_(0,0.01)
 
initNetParams(model)


best=1600
bestmae=100
#global LaplacianMat,UILaplacianMat
print("Uww start")

print("dataset=%s,methord=%s,epoch=%d,batchsize=%d,lr=%.4f,seed=%d,lysz=%d,utopk=%d,itopk=%d,embed_size=%d,info=%s\n"
    %(DATASET,methord,para['epoch'],para['batch_size'],para['lr'],SEED,lysz,utopk,itopk,embed_size,args.info+str(args.dk) ))       

global um,im
for i in range(para['epoch']):


    # if i > para['epoch']*0.75:
    #     optim.param_groups[0]["lr"]= float(para['lr'])*0.1
              
    for id,batch in enumerate(dl):
        # print('epoch:',i,' batch:',id)
        optim.zero_grad()
        prediction,userEmbd,itemEmbd,finalEmbd = model(batch[0].cuda(), batch[1].cuda())
        loss = lossfn(batch[2].float().cuda(),prediction)
        for param in model.parameters():
            loss =loss+0.0001 * param.norm(2)**2 
        loss.backward()
        optim.step()

    
    print(methord,i,' loss ',loss.item()) 
    if i>int(para['epoch'])*args.dk:
        LaplacianMat,UILaplacianMat=model.syn_buildLaplacianMat(rt,finalEmbd=finalEmbd,uuflg=Ruu,iiflg=Rii)
           
        for data in testdl:
            prediction,userEmbd,itemEmbd,finalEmbd2 = model(data[0].cuda(),data[1].cuda())


        loss = lossfn(data[2].float().cuda(),prediction)
         
        if loss<best:
            best=loss
            print("UI epoch best RMSE:",i,para['lr'],itopk,np.sqrt(best.item()) )
            mae=  np.mean(np.abs(data[2].detach().numpy()-prediction.cpu().detach().numpy()))

        
 
print("dataset=%s,methord=%s,epoch=%d,batchsize=%d,lr=%.4f,seed=%d,lysz=%d,utopk=%d,itopk=%d,%.4f,%.4f,info=%s\n"
    %(DATASET,methord,para['epoch'],para['batch_size'],para['lr'],SEED,lysz,utopk,itopk,np.sqrt(best.item()),bestmae,args.info+str(args.dk)+"-"+str(args.decay)))  
  
 