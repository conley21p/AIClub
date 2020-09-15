# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 17:54:41 2020

@author: conle
"""
import random
#import math
#[0,4,7,5,2,6,1,3]

population = 400
main = []
answer = False
generations = 0
#fitnessList = []
vals = []
test =[0,4,7,5,2,6,1,3]
result =[]



def fitness(array):
    #fitnessList = []
    for p in range(population):
        fitness = 0
        for r in range(0,8):
            for c in range(0,8):
                if(fitness == 0):
                    if(r!=c):
                        sum = r + array[p][r]
                        sub= (c-r)*2
                        sum2 = array[p][c] + c -sub
                        if(array[p][r]==array[p][c] or sum == sum2 or sum == array[p][c] + c):
                            #correct = False
                            fitness = fitness +1
                            break;
        if(fitness == 0 ):
            print()
            print("Answerrrrrrrrrrrrrrrrrrr!!!!!!!!!!!!")
            print()
            #result *= array[p]
            print(array[p])
            return True


                
def randomEight(array,count):
    rando = random.randint(0,7)
    if count != 0:
        for i in range(0,count):
            if (array[i] == rando):
                return randomEight(array,count)
    return rando
    

for i in range(population):
    temp =[]
    count = 0
    for a in range(8):
        temp.append(randomEight(temp,count))
        count += 1
    main.append(temp)
#print(main)
answer = fitness(main)

while answer != True:
    mutations = 0
    #print(main[0])
    
    if(answer ==True):
        break
    kids= []
    for _ in range(400):
        #step 1
        a = main[random.randint(0, population-1)]
        b = main[random.randint(0, population-1)]
        #step 2
        spilt = random.randint(0,7)
        #step 3
        child = a[:spilt] + b[spilt:]
        #step 4
        if(random.randint(0,100)<20):
            child[random.randint(0,7)] = random.randint(0,7)
            mutations += 1
        #child = noSameRow(child)
        kids.append(child)
        #kids.append([0,4,7,5,2,6,1,3])
    if(answer ==True):
        break
    generations += 1
    main= kids
    answer = fitness(main)
    #print(generations)
    #print("Gnerationnnnnnn: ", generations)

   
   #print("mutations: ", mutations)  
print("it took ", generations, " Generations")



def noSameRow(child):
    for a in range(0,7):
        for b in range(7,0):
            if(a == b):
                child[b]= randomEight(child,8)
    return child
