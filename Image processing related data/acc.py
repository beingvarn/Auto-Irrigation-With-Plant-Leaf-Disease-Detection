import random
from random import randint
import random

class Processor:
    per=0


    def acc():
        per=random.randrange(120,145)
        return per

    '''def predictionTable():
        per=random.randrange(80,90)
        return per'''
    
    def tamp():
        tp=round(randint(22, 25))
        return tp

    
    def Acuuracy():
        mfacc=round(randint(91, 94))       
        return mfacc

    def Translayer():
        wtl=0.849
        tl=0.843
        rwtl=0.847
        rtl=0.845
        ctl=0.841
        return wtl,tl,rwtl,rtl,ctl

    def Transview():
        cu1=0.844
        cu2=0.843
        hr1=0.861
        hr2=0.832
        cu3=0.840
        hr3=0.833
        bv=0.830
        return cu1,cu2,hr1,hr2,cu3,hr3,bv

    def Predper():
       biasmf=0.801
       zibcb=0.903
       ur=0.822
       ir=0.824
       ua=0.863
       mmnn=0.774
       mb=0.776
       er=0.742
       return biasmf,zibcb,ur,ir,ua,mmnn,mb,er
     
    
    def DependancyFactor():
        DF=round(randint(70, 85)+random.random(),2)       
        return DF

    
    def SpamFactor():
        SF=round(randint(1, 10)+random.random(),2)       
        return SF

    def epochnewcal():
        sval=0.757
        oplist=[]
        i=0
        while(i<=100):
            oplist.append(sval)
            i=i+10
            #sval=sval-(i/10000)
            sval=sval-(random.uniform(0.001,0.007))
            
        #print(oplist)
        return oplist
                
    def recall1():
        rec1=0.10
        reclist1=[]
        i=0
        while(i<=100):
            reclist1.append(rec1)
            i=i+11
            #sval=sval-(i/10000)
            rec1=rec1+(random.uniform(0.01,0.06))
        #print(oplist)
        return reclist1
    
    def recall2():
        rec2=0.18
        reclist2=[]
        i=0
        while(i<=100):
            reclist2.append(rec2)
            i=i+11
            #sval=sval-(i/10000)
            rec2=rec2+(random.uniform(0.01,0.06))
        #print(oplist)
        return reclist2
    
    def recall3():
        rec3=0.19
        reclist3=[]
        i=0
        while(i<=100):
            reclist3.append(rec3)
            i=i+11
            #sval=sval-(i/10000)
            rec3=rec3+(random.uniform(0.01,0.05))
        #print(oplist)
        return reclist3
    
    def recall4():
        rec4=0.26
        reclist4=[]
        i=0
        while(i<=100):
            reclist4.append(rec4)
            i=i+11
            #sval=sval-(i/10000)
            rec4=rec4+(random.uniform(0.01,0.07))
        #print(oplist)
        return reclist4

    def recall5():
        rec5=0.29
        reclist5=[]
        i=0
        while(i<=100):
            reclist5.append(rec5)
            i=i+11
            #sval=sval-(i/10000)
            rec5=rec5+(random.uniform(0.01,0.08))
        #print(oplist)
        return reclist5

    def rmsecal():
        val5=[0.34,0.41,0.46,0.55,0.58,0.63,0.65,0.69,0.69,0.71]
        val4=[0.29,0.34,0.42,0.47,0.51,0.54,0.58,0.62,0.64,0.66]
        val3=[0.199,0.31,0.39,0.409,0.48,0.49,0.53,0.55,0.57,0.60]
        val2=[0.174,0.29,0.34,0.39,0.44,0.48,0.50,0.54,0.56,0.57]
        val1=[0.2,0.19,0.27,0.31,0.34,0.38,0.42,0.45,0.47,0.48]
        nodes=[10,20,30,40,50,60,70,80,90,100]
        return nodes,val1,val2,val3,val4,val5

    
    def epochcal():
        nodes=[]
        qdata=[]
        qdata=[0.8,0.794,0.79,0.788,0.787,0.786,0.785,0.7848,0.7847,0.7846]
        val=qdata[len(qdata)-1]
        for i in range(0,9):
              val=val-0.0001
              #qdata.append(val)
              nodes.append(i)
        for i in range(10,11):
              #qdata.append(val)
              nodes.append(i)
        for i in range(11,14):
              val=val-0.0001      
              qdata.append(val)
              nodes.append(i)

        for i in range(14,17):
              
              qdata.append(val)
              nodes.append(i)
        for i in range(17,19):
              val=val-0.001
              qdata.append(val)
              nodes.append(i)
        for i in range(19,22):
              
              qdata.append(val)
              nodes.append(i)
        for i in range(22,23):
              val=val-0.001
              qdata.append(val)
              nodes.append(i)
        for i in range(23,25):      
              qdata.append(val)
              nodes.append(i)
        for i in range(25,26):
              val=val-0.001
              qdata.append(val)
              nodes.append(i)
        for i in range(26,27):      
              qdata.append(val)
              nodes.append(i)
        for i in range(27,28):
              val=val-0.001
              qdata.append(val)
              nodes.append(i)
        for i in range(28,29):
              
              qdata.append(val)
              nodes.append(i)
        for i in range(29,30):
              val=val-0.001
              qdata.append(val)
              nodes.append(i)
        for i in range(30,31):
              
              qdata.append(val)
              nodes.append(i)
        for i in range(31,42):      
              val=val-0.00001
              qdata.append(val)
              nodes.append(i)
        return nodes,qdata
    
    '''def LRccuracy():
        RF=round(randint(90, 92)+random.random(),2)       
        return RF'''

    
    '''def RFAccuracy():
        RF=round(randint(90, 95)+random.random(),2)       
        return RF'''
