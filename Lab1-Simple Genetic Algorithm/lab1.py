#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author：Janet Chou
import random
import math
import argparse



def fitness_function(tournament_list):
    sum_list = list()
    for m in range(k):
        a = tournament_list[m]
        a = str(a)
        a = list(map(int, a))
        fitness_value = sum(a)
        sum_list.append(fitness_value)
    output_z = tournament_list[sum_list.index(max(sum_list))]
    # output_z=str(output_z)
    return output_z



def mutation(bits_x,chi,n):
    bits_x = list(bits_x)
    mutation_rate = chi / n
    output_z = bits_x[:]
    # print(bits_x)
    for j in range(n):
        random_mutation_probability = random.random()
        if random_mutation_probability <= mutation_rate:
            if output_z[j] == '0':
                output_z[j] = '1'
            else:
                output_z[j] = '0'
                # print(bits_x[j])
    # print(''.join(output_z))
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
    # print(z)
    return z


def tournament_selection(origin_population,k):
    tournament_list = list()
    # print(tournament_list)
    # random_index = list(range(population_size))
    # print(random_index)
    random_list = [random.randint(0, population_size - 1) for z in range(k)]
    # print(random_list)
    for l in random_list:
        tournament_list.append(origin_population[l])
    # print(tournament_list)
    return fitness_function(tournament_list)


def encoding(population_size,n):
    origin_population=list()
    upper_bound=math.pow(2,n)
    for i in range(population_size):
        bitstring_number=int(random.randint(0,upper_bound))
        bitstring='{0:b}'.format(bitstring_number)
        bitstring_number=bitstring.zfill(n)
        origin_population.append(bitstring_number)
    # print(origin_population)
    return origin_population


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('-question',type=int,default=4)
    parser.add_argument('-bits_x', type=str, default='11000')
    parser.add_argument('-bits_y', type=str, default='11111')
    parser.add_argument('-chi', type=float, default=0.2)
    parser.add_argument('-repetitions', type=int, default=4)
    parser.add_argument('-population', type=str, default='0000 1101 1000 1100 1111 1010')
    parser.add_argument('-k', type=int, default=2)
    parser.add_argument('-n', type=int, default=10)
    parser.add_argument('-lamda','-lambda', type=int, default=10)

    args= parser.parse_args()
    question_number=args.question
    if question_number==1:
        try:
            # bits_x,chi,repetitions=input('Exercise 1 please input bitstring,chi and repetitions respectively'
            #                              '（use space to separate them）：').split('-')
            bits_x=args.bits_x
            n = len(bits_x)
            chi = args.chi
            repetitions=args.repetitions
            if chi < 0 or chi > n:
                print('chi must be greater than or equal to zero and not greater than the length of bits_x')
            else:
                repetitions = int(repetitions)
                for i in range(repetitions):
                    result_1 = (mutation(bits_x, chi, n))
                    print(''.join(result_1))
        except Exception as e:
            print('you enter the wrong information,please start again')

    elif question_number==2:
        try:
            # bits_x,bits_y,repetitions = input('Exercise 2 please input bits_x,bits_y and repetitions respectively'
            #                                  '（use space to separate them）：').split('-')
            bits_x=args.bits_x
            bits_y=args.bits_y
            repetitions=args.repetitions
            n_x = len(bits_x)
            n_y = len(bits_y)
            n=n_x
            if n_x != n_y:
                print('bits_x and bits_y must be the same length,please start again')
            else:
                bits_x = list(bits_x)
                bits_y = list(bits_y)
                repetitions = int(repetitions)
                for i in range(repetitions):
                    result_2=crossover(bits_x,bits_y,n)
                    print(''.join(result_2))
        except Exception as e:
            print('you enter the wrong information,please start again')


    elif question_number == 3:
        try:
            # bits_x= input('Exercise 3 please input bits_x("0"or"1")')
            bits_x = args.bits_x
            n = len(bits_x)
            bits_x = list(map(int, bits_x))
            result_3 = sum(bits_x)
            print(result_3)
        except Exception as e:
            print('you enter the wrong information,please start again')

    elif question_number == 4:
        try:
            # population_bitstrings= input('Exercise 4 please input population_bitstrings（every bitstring must have the same length '
            #                              'and use space to separate them）：')
            # k,repetition=input('please input tournament size and repetitions（use space to separate them）：').split('-')
            population_bitstrings =args.population

            k=args.k
            repetition=args.repetitions
            population_bitstrings = list(population_bitstrings.split(' '))
            # population_bitstrings = list(map(int, population_bitstrings))
            # print(population_bitstrings)
            population_size = len(population_bitstrings)
            # repetition = int(repetition)
            for j in range(repetition):
                result_4=tournament_selection(population_bitstrings,k)
                print(result_4)

        except Exception as e:
            print('you enter the wrong information,please start again')





    else:
        try:
            # chi,n,population_size,k,repetitions=input('Exercise 5 please input chi,bitstring length,population size,tournament size'
            #                                           ' and repetitions respectively（use space to split them）：').split('-')

            chi = args.chi
            n = args.n
            k = args.k
            population_size = args.lamda
            repetitions =args.repetitions

            fbest = n
            # print(n)



            upper_bounder = int(math.pow(2, n)) - 1

            # print(upper_bounder)
            xbest = '{0:b}'.format(upper_bounder)
            # print(xbest)

            origin_population = encoding(population_size, n)
            population = origin_population[:]
            origin_population_result = origin_population[:]
            # print(origin_population)



            # 最后生成4个结果
            for j in range(repetitions):
                population = origin_population_result[:]
                # print(population)
                t = 0
                z = str()
                flag = True
                while flag:
                    origin_population = population[:]
                    # print(origin_population)
                    population = list()
                    t = t + 1
                    # print(t)
                    for h in range(population_size):
                        x = tournament_selection(origin_population, k)
                        y = tournament_selection(origin_population, k)
                        z = crossover(mutation(x, chi, n), mutation(y, chi, n), n)
                        # print(z)
                        if z == xbest:
                            # print(z)
                            flag = False
                        population.append(z)
                print(str(n) + "\t" + str(chi) + "\t" + str(population_size) + "\t" + str(k) + "\t" + str(
                    t) + "\t" + str(fbest) + "\t" + xbest)

        except Exception as e:
            print('you enter the wrong information,please start again')

