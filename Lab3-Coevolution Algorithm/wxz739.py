#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author：Janet Chou
import numpy as np
import time
import argparse
from scipy import stats


def Sample(prob):
    n = len(prob)
    integers = np.arange(n)
    all_prob = sum(prob)
    total_prob = np.cumsum(prob)
    dics = np.random.uniform(0, all_prob)
    for i in range(n):
        if dics <= total_prob[i]:
            return integers[i]


def strategy_handle(strategy):
    attendance_prob = np.array([])
    A = np.array([])
    B = np.array([])
    for i in range(h):
        attendance_prob = np.append(attendance_prob, strategy[i * (2 * h + 1)])
        A = np.concatenate((A, strategy[1 + i * (2 * h + 1):h + 1 + i * (2 * h + 1)]))
        B = np.concatenate((B, strategy[h + 1 + i * (2 * h + 1):2 * h + 1 + i * (2 * h + 1)]))
    A = A.reshape(h, h)
    B = B.reshape(h, h)
    return attendance_prob,A,B


def one_step_desicion(state,crowded,attendance_prob,A,B):
    if crowded:
        next_state=Sample(A[state])
    else:
        next_state=Sample(B[state])
    if np.random.random()<=attendance_prob[next_state]:
        decision=1
    else:
        decision = 0
    return decision,next_state


def all_decisions(origin_population):
    decision_list=list()
    for i in origin_population:
        state=i[0]
        P=i[1]
        if np.random.random() <= P[state]:
            decision_list.append(1)
        else:
            decision_list.append(0)
    bar_people=sum(decision_list)
    return bar_people,decision_list


def initial_individual(h):
    state=np.random.randint(0,h)
    P=np.random.random(h)
    A=np.zeros((h,h))
    B=np.zeros((h,h))
    for i in range(h):
        random_list=np.random.randint(0,100,h)
        Sum=sum(random_list)
        A[i]=random_list/Sum
    for j in range(h):
        random_list = np.random.randint(0, 100, h)
        Sum = sum(random_list)
        B[j] = random_list / Sum
    return state,P,A,B


def initial_population(population_size,h):
    origin_population=list()
    for i in range(population_size):
        strategy = list(initial_individual(h))
        origin_population.append(strategy)
    return origin_population


def fitness_function(weeks,origin_population,population_size):
    payoff_list=[0]*population_size
    for i in range(0,weeks):
        bar_people, decision_list = all_decisions(origin_population)
        if bar_people<0.6*population_size:
            crowded=0
        else:
            crowded=1
        for c in range(population_size):
            if decision_list[c]!=crowded:
                payoff_list[c]+=1
        for m in range(population_size):
            # print(origin_population[i][0])
            # print('----------------------------')
            A=origin_population[m][2]
            B=origin_population[m][3]
            if crowded:
                origin_population[m][0]=Sample(A[origin_population[m][0]])
            else:
                origin_population[m][0] = Sample(B[origin_population[m][0]])
            # print(origin_population[i][0])
        decision_list = list(map(str, decision_list))
        print(str(i)+'\t'+str(t)+'\t'+str(bar_people)+'\t'+str(crowded)+'\t'+('\t'.join(decision_list)))
    return payoff_list,origin_population


def Roulette_Selection(payoff,copy_origin_population):
    Sum_payoff=sum(payoff)
    payoff_array=np.array(payoff)
    payoff_distribution=payoff_array/Sum_payoff
    select_index=Sample(payoff_distribution)
    # print(select_index)
    output=copy_origin_population[select_index]
    return output

def mutation(x,h,mutation_rate):
    strategy = list(initial_individual(h))
    for i in range(4):
        if np.random.random()<=mutation_rate:
            x[i]=strategy[i]
    return x

def crossover(x,y,h):
    cross_locus=np.random.randint(h/2-2,h/2+2)
    z1=x[:]
    z2=y[:]
    for i in range(1, 4):
        z1[i]=x[i].copy()
        z2[i]=y[i].copy()
    # print(cross_locus)
    z1[1][cross_locus:h] = y[1][cross_locus:h].copy()
    z2[1][cross_locus:h] = x[1][cross_locus:h].copy()
    for i in range(2,4):
        z1[i][cross_locus:h,:]=y[i][cross_locus:h,:].copy()
        z2[i][cross_locus:h,:]=x[i][cross_locus:h,:].copy()
    return z1,z2



if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='manual to this script',add_help=False)
    parser.add_argument('-question',type=int,default=3)
    parser.add_argument('-repetitions', type=int, default=5)
    parser.add_argument('-prob', type=str, default='0 0 1 0')
    parser.add_argument('-strategy', type=str, default='2 0.1 0.0 1.0 1.0 0.0 1.0 0.9 0.1 0.9 0.1')
    parser.add_argument('-state',type=int,default=1)
    parser.add_argument('-crowded', type=int, default=0)
    parser.add_argument('-lamda','-lambda', type=int, default=700)
    parser.add_argument('-h',type=int,default=10)
    parser.add_argument('-weeks', type=int, default=10)
    parser.add_argument('-max_t', type=int, default=20)
    args= parser.parse_args()
    question_number=args.question

    if question_number==1:
        prob = list(args.prob.split(' '))
        prob = list(map(float, prob))
        for i in range(args.repetitions):
            output_1 = Sample(prob)
            print(output_1)

    elif question_number == 2:
        strategy=list(map(eval,args.strategy.split(' ')))
        state=args.state
        crowded=args.crowded
        h=int(strategy[0])
        strategy=strategy[1:len(strategy)]
        attendance_prob, A, B=strategy_handle(strategy)
        for i in range(args.repetitions):
            d,s=one_step_desicion(state,crowded,attendance_prob,A,B)
            print(str(d)+'\t'+str(s))


    else:

        population_size = args.lamda
        h=args.h
        weeks=args.weeks
        time_budget=args.max_t

        mutation_rate=0.0001
        t=0
        origin_population=initial_population(population_size,h)
        start = time.clock()
        while True:
            copy_origin_population = origin_population[:]
            for i in range(population_size):
                copy_origin_population[i] = origin_population[i][:]
            end = time.clock()
            runtime = end - start
            # print(runtime)
            if t== time_budget-1:
                # print(runtime)
                break
            population=list()

            payoff, origin_population = fitness_function(weeks, origin_population, population_size)
            # print(bar_people)
            # Sum_payoff=sum(payoff)
            # print(Sum_payoff)
            t = t + 1
            for j in range(int(population_size/2)):
                father=Roulette_Selection(payoff,copy_origin_population)
                mother=Roulette_Selection(payoff,copy_origin_population)
                x=mutation(father,h,mutation_rate)
                y=mutation(mother,h,mutation_rate)
                z1,z2=crossover(x,y,h)
                population.append(z1)
                population.append(z2)
                middle=time.clock()
                # if (middle-start)>time_budget:
                #     print(runtime)
                #     break
            origin_population=population[:]
            # print(len(origin_population))
            # print("-----")
        final_payoff, origin_population=fitness_function(weeks, origin_population, population_size)
        # print("-----")
        # final_population=list()
        # for h in range(population_size):
        #     best= Roulette_Selection(final_payoff,copy_origin_population)
        #     final_population.append(best)
        # final_payoff, origin_population, bar_people = fitness_function(weeks, final_population, population_size)
        # print(bar_people)
        # final_payoff=sum(final_payoff)
        # print(final_payoff)
        # print(t)





