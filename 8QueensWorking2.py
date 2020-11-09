"""
Created on Mon Sep  7 17:54:41 2020
@author: conle
Goal: Program will solve the 8 queens chess problem by random evolution of the population
The 8 queens problem is trying to find a state of a chess board where if there were 8 queens
on the board none of the queens could attack the other
"""
import random

population = 400
main = []
answer = False
generations = 0
vals = []
test =[0,4,7,5,2,6,1,3]
result =[]

#Find the fitness of board and or if its correct
def fitness(array):
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
            print(array[p])
            return True

#Method make sures a array has all seperate value 0-7          
def randomEight(array,count):
    rando = random.randint(0,7)
    if count != 0:
        for i in range(0,count):
            if (array[i] == rando):
                return randomEight(array,count)
    return rando
    
#Create intial population of boards
for i in range(population):
    temp =[]
    count = 0
    for a in range(8):
        temp.append(randomEight(temp,count))
        count += 1
    main.append(temp)
answer = fitness(main)

#Main loop that evolves the boards randomly within each other till answer is found
while answer != True:
    mutations = 0
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
        kids.append(child)
    if(answer ==True):
        break
    generations += 1
    main= kids
    answer = fitness(main)

#Output the generation took to find answer
print("it took ", generations, " Generations")

