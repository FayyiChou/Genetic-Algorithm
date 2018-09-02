#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author：Janet Chou
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author：Janet Chou
from random import random, randint, choice
from copy import deepcopy
import numpy as np
import time
from math import log
import argparse
import fileinput
import signal

def parse_expression(n,x,expression):
    while '(' in expression:
        origin = expression
        for i in range(len(expression)):
            if origin!=expression:
                break
            else:
                if expression[i]==')':
                    sub_expr=expression[:i+1]
                    for j in range(len(sub_expr)-1,-1,-1):
                        if sub_expr[j]=='(':
                            replace_sub=sub_expr[j:]
                            useful_expr=sub_expr[j+1:-1]
                            result=functions(n,x,useful_expr)
                            expression=expression.replace(replace_sub,result)
                            break
    return expression

def functions(n,x,useful_expr):
    useful_expr = list(useful_expr.split(' '))
    if useful_expr[0]=='add':
        result=np.add(float(useful_expr[1]),float(useful_expr[2]))
        return str(result)
    elif useful_expr[0]=='sub':
        result=np.subtract(float(useful_expr[1]),float(useful_expr[2]))
        return str(result)
    elif useful_expr[0] == 'mul':
        result=np.multiply(float(useful_expr[1]),float(useful_expr[2]))
        return str(result)
    elif useful_expr[0]=='div':
        if float(useful_expr[2])==0:
            return str(0)
        else:
            result=np.divide(float(useful_expr[1]),float(useful_expr[2]))
            return str(result)
    elif useful_expr[0] == 'pow':
        if np.isnan(np.power(float(useful_expr[1]),float(useful_expr[2]))):
            return str(0)
        else:
            result=np.power(float(useful_expr[1]),float(useful_expr[2]))
            return str(result)
    elif useful_expr[0]=='sqrt':
        if float(useful_expr[1])<=0:
            return str(0)
        else:
            result=np.sqrt(float(useful_expr[1]))
            return str(result)

    elif useful_expr[0] == 'log':
        if float(useful_expr[1])<=0:
            return str(0)
        else:
            result=np.log2(float(useful_expr[1]))
            return str(result)

    elif useful_expr[0]=='exp':
        if np.isnan(np.exp(float(useful_expr[1]))):
            return str(0)
        else:
            result=np.exp(float(useful_expr[1]))
            return str(result)
    elif useful_expr[0] == 'max':
        result=np.maximum(float(useful_expr[1]),float(useful_expr[2]))
        return str(result)
    elif useful_expr[0]=='ifleq':
        if float(useful_expr[1])<= float(useful_expr[2]):
            return str(useful_expr[3])
        else:
            return str(useful_expr[4])

    elif useful_expr[0] == 'data':
        if np.isnan(float(useful_expr[1])):
            return str(float(x[0]))
        elif np.isinf(float(useful_expr[1])):
            return str(float(x[0]))
        else:
            index_x=np.mod(abs(np.floor(float(useful_expr[1]))),n)
            result=x[int(index_x)]
            return str(result)

    elif useful_expr[0]=='diff':
        float_value1=float(useful_expr[1])
        float_value2=float(useful_expr[2])
        if np.isnan(float_value1):
            k=0
        elif np.isinf(float_value1):
            k= 0
        else:
            k = np.mod(int(abs(np.floor(float(useful_expr[1])))), n)
        if np.isnan(float_value2):
            l=0
        elif np.isinf(float_value2):
            l= 0
        else:
            l=np.mod(int(abs(np.floor(float(useful_expr[2])))),n)
        result=np.subtract(x[int(k)],x[int(l)])
        return str(result)
    elif useful_expr[0] == 'avg':
        float_value1 = float(useful_expr[1])
        float_value2 = float(useful_expr[2])
        if np.isnan(float_value1):
            k=0
        elif np.isinf(float_value1):
            k=0
        else:
            k = np.mod(int(abs(np.floor(float(useful_expr[1])))), n)
        if np.isnan(float_value2):
            l=0
        elif np.isinf(float_value2):
            l=0
        else:
            l = np.mod(int(abs(np.floor(float(useful_expr[2])))), n)
        # sum_x=0
        if k==l:
            return str(0)
        elif k<l:
            return str(np.mean(list(map(float,x[k:l]))))
        else:
            return str(np.mean(list(map(float, x[l:k]))))
            # t=np.minimum(k,l)
            # upper=np.maximum(k,l)-1
            # for z in range(int(t),int(upper)+1):
            #     sum_x=sum_x+x[z]
            # result=np.divide(sum_x,abs(k-l))
            # return str(result)

def fitness_function(filename,m,n,expression):
    calculator=list()
    with open(filename,'r') as f:
        for i in f:
            # print(i)
            train_data=list(map(float,i.strip('\n').strip('\t').split('\t')))
            x=train_data[:n]
            y=train_data[n]
            output = float(parse_expression(n, x, expression))
            # print(output)
            loss_value=(output-y)**2
            calculator.append(loss_value)
    # print(sum(calculator))
    avarge_loss=np.divide(sum(calculator),m)
    return avarge_loss


class fwrapper:
    def __init__(self,function,childcount,name):
        self.function=function
        self.childcount=childcount
        self.name=name

class node:
    def __init__(self,fw,children):
        self.function=fw.function
        self.name=fw.name
        self.children=children

    def evaluate(self,inp):
        results=[n.evaluate(inp) for n in self.children]
        return self.function(results)
    # def display(self,indent=0):
    #     print (' ('+self.name,end='')
    #     # print ('('+(' '*indent)+self.name)
    #     for c in self.children:
    #         c.display(indent+1)
    #     print (')',end="")
    def get_functionStr(self):
        output = ''
        output = output+' ('+self.name
        for c in self.children:
            output = output + c.get_functionStr()
        output = output+')'

        return output


class data_node:
    def __init__(self,idx):
        self.idx=idx

    def evaluate(self,inp):
        return inp[self.idx]

    # def display(self,indent=0):
    #     # print('%sdata%d' % (' '*indent,self.idx))
    #     print(' (data '+str(self.idx)+')',end="")

    def get_functionStr(self):
        output = ' (data '+str(self.idx)+')'
        return output

class data_diff_node:
    def __init__(self,e1,e2):
        self.e1 = e1
        self.e2 = e2
    def evaluate(self,inp):
        return inp[self.e1]-inp[self.e2]

    # def display(self,indent=0):
    #     # print('%sdata%d' % (' '*indent,self.e1),end="")
    #     print(' (diff '+str(self.e1)+' '+str(self.e2)+')',end='')

    def get_functionStr(self):
        output = ' (diff '+str(self.e1)+' '+str(self.e2)+')'
        return output

class data_avg_node:
    def __init__(self,e1,e2):
        self.e1 = e1
        self.e2 = e2
    def evaluate(self,inp):
        if self.e2 == self.e1:
            return 0
        elif self.e2 > self.e1:return np.mean(list(inp[self.e1:self.e2]))
        else: return np.mean(list(inp[self.e2:self.e1]))

    # def display(self,indent=0):
    #     # print('%sdata%d' % (' '*indent,self.e1),end="")
    #     print(' (avg '+str(self.e1)+' '+str(self.e2)+')',end="")

    def get_functionStr(self):
        output = ' (avg '+str(self.e1)+' '+str(self.e2)+')'
        return output


class constnode:
    def __init__(self,v):
        self.v=v
    def evaluate(self,inp):
        return self.v
    # def display(self,indent=0):
    #     print(' '+str(self.v),end='')
    #     # print('%s%d' % (' '*indent,self.v))

    def get_functionStr(self):
        output = ' '+str(self.v)
        return output

def makerandomtree(n, maxdepth=3, fpr=0.5, ppr=0.6, p_diff=0.6, p_avg=0.6):
    if random() < fpr and maxdepth > 0:
        f = choice(flist)
        children = [makerandomtree(n, maxdepth - 1, fpr, ppr) for i in range(f.childcount)]
        return node(f, children)
    elif random() < ppr:
        return data_node(randint(0, n - 1))
    elif random() < p_diff:
        return data_diff_node(randint(0, n - 1), randint(0, n - 1))
    elif random() < p_avg:
        return data_avg_node(randint(0, n - 1), randint(0, n - 1))
    else:
        # return constnode(randint(0,len(x)))
        # use 0-10
        return constnode(randint(0, 10))

def scorefunction(tree, dataset, n):
    dif = 0
    for single_data in dataset:
        y = single_data[-1]

        v = tree.evaluate(single_data[0:n])
        #
        dif += abs(v - y)
        #
    return dif


def mutate(t,pc,probchange=0.1):
    if random()<probchange:
        return makerandomtree(pc)
    else:
        result=deepcopy(t)
        if hasattr(t,"children"):
            result.children=[mutate(c,pc,probchange) for c in t.children]
    return result

def crossover(t1,t2,probswap=0.7,top=1):
    if random() < probswap and not top:
        return deepcopy(t2)
    else:
        result=deepcopy(t1)
        if hasattr(t1,'children') and hasattr(t2,'children'):
            result.children=[crossover(c,choice(t2.children),probswap,0) for c in t1.children]
        return result


def getrankfunction(dataset, n):
    def rankfunction(population):
        scores = [(scorefunction(t, dataset, n), t) for t in population]
        # print('scores=',scores)
        scores.sort(key=lambda s: s[0])
        return scores

    return rankfunction

def set_timeout(num, callback):
    def wrap(func):
        def handle(signum, frame):  #
            raise RuntimeError
        def to_do(*args, **kwargs):
            try:
                signal.signal(signal.SIGALRM, handle)  #
                signal.alarm(num)  #
                r = func(*args, **kwargs)
                signal.alarm(0)  #
                return r
            except RuntimeError as e:
                callback()
        return to_do
    return wrap


def evolve(pc, popsize, time_budget, file_data, maxgen=500, mutationrate=0.1, breedingrate=0.4, pexp=0.7, pnew=0.05):
    time_start = time.clock()
    dataset = []
    for single_line in fileinput.input([file_data]):
        x = single_line.split('\t')
        x = map(float, x)
        dataset.append(x)

    rankfunction = getrankfunction(dataset, pc)

    def selectindex():
        return int(log(random()) / log(pexp))

    population = [makerandomtree(pc) for i in range(popsize)]

    for i in range(maxgen):
        scores = rankfunction(population)

        time_end = time.clock()
        if time_end - time_start >= time_budget:
            return scores[0][1].get_functionStr().strip()

        newpop = [scores[0][1], scores[1][1]]

        while len(newpop) < popsize:

            if random() > pnew:
                newpop.append(
                    mutate(crossover(scores[selectindex()][1], scores[selectindex()][1], probswap=breedingrate), pc,
                           probchange=mutationrate))
            else:
                newpop.append(makerandomtree(pc))

        population = newpop

#############################################################

def function_div(l):
    if l[1] == 0: return 0
    else:
        return l[0]/l[1]

def function_pow(l):
    if np.isnan(np.power(float(l[0]), l[1])):
        return 0
    else: return np.power(float(l[0]), l[1])

def function_sqrt(l):
    if l[0] <= 0:
        return 0
    else:
        return np.power(l[0],0.5)

def function_log(l):

    if l[0] <= 0:return 0
    else:return np.log2(l[0])

def function_exp(l):

    if np.isnan(np.exp(l[0])):
            return 0
    else : return np.exp(l[0])


def function_ifleq(l):
    if l[0] <= l[1]:return l[2]
    else:return l[3]


f_add=fwrapper(lambda l:l[0] + l[1],2,'add')
f_sub=fwrapper(lambda l:l[0] - l[1],2,'sub')
f_mul=fwrapper(lambda l:l[0] * l[1],2,'mul')
f_div=fwrapper(function_div,2,'div')
f_pow=fwrapper(function_pow,2,'pow')
f_sqrt=fwrapper(function_sqrt,1,'sqrt')
f_log=fwrapper(function_log,1,'log')
f_exp=fwrapper(function_exp,1,'exp')
f_max=fwrapper(lambda l:max(l[0],l[1]),2,'max')
f_ifleq=fwrapper(function_ifleq,4,'ifleq')
flist=[f_add,f_sub,f_mul,f_div,f_pow,f_sqrt,f_log,f_exp,f_max,f_ifleq]

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('-question', type=int, default=3)
    parser.add_argument('-n',type=int,default=10)
    parser.add_argument('-x', type=str, default='1.0 2.0')
    parser.add_argument('-expr', type=str, default='(mul (add 1 2) (log 8))')
    parser.add_argument('-m', type=int, default=10)
    parser.add_argument('-data',default='data.txt')
    parser.add_argument('-time_budget', type=int, default=10)
    parser.add_argument('-lamda','-lambda', type=int, default=700)
    args= parser.parse_args()
    question_number=args.question

    if question_number==1:
        n=args.n
        x=list(map(float,args.x.split(' ')))
        expression=args.expr
        result1=parse_expression(n, x, expression)
        print(result1)
    elif question_number == 2:
        expression = args.expr
        n=args.n
        m=args.m
        filename=args.data
        result2=fitness_function(filename,m,n,expression)
        print(result2)
    else:
        m=args.m
        n=args.n
        population_size=args.lamda
        time_budget=args.time_budget
        print(evolve(n, population_size, time_budget, args.data, maxgen=500, mutationrate=0.1, breedingrate=0.4,
                   pexp=0.7, pnew=0.05))