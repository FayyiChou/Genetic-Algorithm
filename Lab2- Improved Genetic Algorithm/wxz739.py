#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author：Janet Chou
import time
import random
import math
import argparse

#read WDIMACS file
def read_file(filename):
    with open(filename,'r') as f:
        context = list(f.readlines())
        count_n = 0
        for i in context:
            count_n += 1
            if i[0] == 'p':
                arguments = i.strip('\n').split(' ')
                variable_number = int(arguments[2])
                clause_number = int(arguments[3])
                clause_list = context[count_n:len(context)]
                return variable_number, clause_number, clause_list


def check_satisafiability(clause,assignment):
    end_index = len(clause) - 1
    clause=clause[1:end_index]
    for i in clause:
        i=int(i)
        if i>0:
            x_index = i - 1
            if assignment[x_index]=='1':
                return 1

        else:
            x_index = abs(i)- 1
            if assignment[x_index]=='0':
                return 1
    else:
        return 0

def traverse_clause_list(clause_list,assignment):
    satisfied_number = 0
    for i in clause_list:
        clause = list(i.strip('\n').split(' '))
        output = check_satisafiability(clause, assignment)
        if output == 1:
            satisfied_number += 1
    return satisfied_number

def initial_population(population_size,variable_number):
    origin_population=list()
    upper_bound=2**variable_number
    for i in range(population_size):
        bitstring_number=int(random.randint(0,upper_bound-1))
        bitstring='{0:b}'.format(bitstring_number)
        bitstring_number=bitstring.zfill(variable_number)
        origin_population.append(bitstring_number)
    # print(origin_population)
    return origin_population

def tournament_selection(origin_population,k):
    tournament_list = list()
    random_index = list(range(population_size))
    random_list = random.sample(random_index, k)
    for l in random_list:
        tournament_list.append(origin_population[l])
    return fitness_function(tournament_list)

def fitness_function(tournament_list):
    sum_list=list()
    for m in tournament_list:
        fitness_value=traverse_clause_list(clause_list,m)
        sum_list.append(fitness_value)
    number_sat=max(sum_list)
    output_z = tournament_list[sum_list.index(number_sat)]
    return output_z

def mutation(bits_x,mutation_rate):
    bits_x = list(bits_x)
    output_z = bits_x[:]
    for j in range(variable_number):
        random_mutation_probability = random.random()
        if random_mutation_probability <= mutation_rate:
            if output_z[j] == '0':
                output_z[j] = '1'
            else:
                output_z[j] = '0'
    return output_z




def crossover(bits_x,bits_y,n):
    output_z = list()
    for j in range(n):
        if bits_x[j] != bits_y[j]:
            if random.random() <= 0.5:
                output_z.append('0')
            else:
                output_z.append('1')
        else:
            output_z.append(bits_x[j])
    z=''.join(output_z)
    return z





if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('-question',type=int,default=3)
    parser.add_argument('-clause', type=str, default='0.5 1 2 -3 -4 0')
    parser.add_argument('-assignment', type=str, default='0000')
    parser.add_argument('-wdimacs', default='3col80_5_2.shuffled.cnf.wcnf')
    parser.add_argument('-repetitions', type=int, default=5)
    parser.add_argument('-time_budget', type=int, default=20)


    args= parser.parse_args()
    question_number=args.question
    if question_number==1:
        clause = list(args.clause.split(' '))
        assignment=args.assignment
        output=check_satisafiability(clause,assignment)
        print(output)


    elif question_number == 2:
        assignment = args.assignment
        filename=args.wdimacs
        variable_number, clause_number, clause_list=read_file(filename)
        output=traverse_clause_list(clause_list,assignment)
        print(output)


    else:
        time_budget = args.time_budget
        repetitions = args.repetitions
        filename = args.wdimacs
        variable_number, clause_number, clause_list = read_file(filename)
        mutation_rate=0.0001
        k=3
        population_size=70
        # nsat=clause_number

        for j in range(repetitions):
            start = time.clock()
            origin_population = initial_population(population_size, variable_number)
            # z_value =0
            t = 0
            z = str()
            flag = True
            while flag:
                population = list()
                population_value=list()

                end = time.clock()
                runtime = end - start
                if runtime>time_budget:
                    break
                t = t + 1
                for h in range(population_size):
                    x = tournament_selection(origin_population, k)
                    y = tournament_selection(origin_population, k)
                    z = crossover(mutation(x,mutation_rate ), mutation(y, mutation_rate), variable_number)
                    # z_value=traverse_clause_list(clause_list,z)
                    population.append(z)
                    middle=time.clock()
                    if (middle-start)>time_budget:
                        break
                    # population_value.append(z_value)
                # z_value=max(population_value)
                # z = population[population_value.index(z_value)]
                origin_population = population[:]
            z = fitness_function(origin_population)
            z_value = traverse_clause_list(clause_list, z)
            print(str(t*population_size)+'\t'+str(z_value)+'\t'+str(z)+'\t'+str(runtime))



