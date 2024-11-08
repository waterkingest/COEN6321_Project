# hyper parameter
import random
import numpy as np
import copy
Rowsize=8
Colsize=8
offspringPercent=0.7
population_size=100
input_puzzle=[]
maxGeneration=100
swap_probability=80
rotate_probability=80
with open('Example_Input&Output\Ass1Input.txt', 'r') as f:
    for line in f:
        input_puzzle.append(line.strip().split(' '))
Code_dictionary={}
code=1
for i in range(1,Rowsize+1):
    for j in range(1,Colsize+1):
        Code_dictionary[str(code)]=input_puzzle[i-1][j-1]
        code+=1
def decode_dictionary(piece):
    return Code_dictionary[piece]

def initionation():
    population=[]
    for i in range(population_size):
        Code_puzzle = random.sample(range(1, Rowsize*Colsize+1), Rowsize*Colsize)
        puzzle = [[x,decode_dictionary(str(x))]for x in Code_puzzle]
        population.append(puzzle)
    return population


def calculatRowMisMatch(puzzleRow1,puzzleRow2):
    numberOfMisMatch=0
    for n,piece in enumerate(puzzleRow1):
        if piece[2]!=puzzleRow2[n][0]:
            numberOfMisMatch+=1
    return numberOfMisMatch
def calculatColMisMatch(puzzleCol1,puzzleCol2):
    numberOfMisMatch=0
    for n,piece in enumerate(puzzleCol1):
        if piece[1]!=puzzleCol2[n][3]:
            numberOfMisMatch+=1
    return numberOfMisMatch

def calculateFitness(puzzle):
    puzzle=[x[1] for x in puzzle]
    puzzle=np.array(puzzle).reshape(Rowsize, Colsize).tolist()
    fitness=0
    for n in range(Rowsize-1):
        fitness+=calculatRowMisMatch(puzzle[n],puzzle[n+1])
    for n in range(Colsize-1):
        col1,col2=[],[]
        for i in range(Rowsize):
            col1.append(puzzle[i][n])
            col2.append(puzzle[i][n+1])
        fitness+=calculatColMisMatch(col1,col2)
    # fitness=-fitness
    return fitness

def mutation1(puzzle):
    '''
    swap two pieces
    '''
    global swap_probability
    probability = random.randint(1, 100)
    if probability <= 100-swap_probability:
        pass
    else:
        swap_numb=random.randint(1,Rowsize*Colsize)
        for _ in range(swap_numb):
            index1,index2=random.sample(range(Rowsize*Colsize),2)
            puzzle[index1],puzzle[index2]=puzzle[index2],puzzle[index1]
    # puzzle=localSearch(puzzle)
    return puzzle
def mutation2(puzzle):
    '''
    rotate a piece
    '''
    global rotate_probability
    probability = random.randint(1, 100)
    if probability <= 100-rotate_probability:
        pass
    else:
        mutation2_numb=random.randint(1,Rowsize*Colsize)
        for _ in range(mutation2_numb):
            rotate_times=random.randint(1,3)
            index=random.randint(0,Rowsize*Colsize-1)
            # print(index,rotate_times)
            puzzle[index][1]=puzzle[index][1][-rotate_times:]+puzzle[index][1][:-rotate_times]
    # puzzle=localSearch(puzzle)
    return puzzle

def reshape(matrix,row,col):
    total_elements = row*col
    if total_elements != len(matrix):
        raise ValueError("Size Error")
    result=[]
    result=[matrix[i:i+col] for i in range(0, len(matrix), col)]
    return result

def flatten(matrix):
    flattened_arr = [item for submatrix in matrix for item in submatrix]
    return flattened_arr

def Crossover(puzzle1,puzzle2,windowrow,windowcol):
    '''
    Order crossover
    '''
    parent1=reshape(puzzle1,Rowsize,Colsize)
    parent2=reshape(puzzle2,Rowsize,Colsize)
    rand_row_index=random.randint(0,Rowsize-windowrow)
    rand_col_index=random.randint(0,Colsize-windowcol)
    part1,part2=[],[]
    record_p1,record_p2=[],[]
    for row in parent1[rand_row_index:rand_row_index+windowrow]:
        part1+=row[rand_col_index:rand_col_index+windowcol]
        record_p1+=[x[0] for x in row[rand_col_index:rand_col_index+windowcol]]
    for row in parent2[rand_row_index:rand_row_index+windowrow]:
        part2+=row[rand_col_index:rand_col_index+windowcol]
        record_p2+=[x[0] for x in row[rand_col_index:rand_col_index+windowcol]]
    c1=np.zeros((Rowsize,Colsize),int).tolist()
    c2=np.zeros((Rowsize,Colsize),int).tolist()
    star=0
    for row_i in range(rand_row_index,rand_row_index+windowrow):
        c1[row_i][rand_col_index:rand_col_index+windowcol]=part1[star:star+windowcol]
        c2[row_i][rand_col_index:rand_col_index+windowcol]=part2[star:star+windowcol]
        star+=windowcol
    c1=flatten(c1)
    c2=flatten(c2)
    c1_index,c2_index=0,0
    for i in range(Rowsize*Colsize):
        if  c1_index<Rowsize*Colsize and c1[c1_index]==0:
            if puzzle1[i][0] not in record_p1:
                c1[c1_index]=puzzle1[i]
                c1_index+=1
            else:
                pass
        else:
            c1_index+=windowcol
        if c2_index<Rowsize*Colsize and c2[c2_index]==0 :
            if puzzle2[i][0] not in record_p2:
                c2[c2_index]=puzzle2[i]
                c2_index+=1
            else:
                pass
        else:
            c2_index+=windowcol
    return c1,c2

def encode_dictionary(puzzle):
    codeDictionary={}
    for i in range(1,Rowsize*Colsize+1):
        codeDictionary[str(puzzle[i-1][0])]=puzzle[i-1][1]
    return codeDictionary
def find_puzzle2_edge(puzzle_list,element):
    index2=puzzle_list.index(element)
    if index2 == Rowsize*Colsize-1:
        return 0,index2-1
    else:
        return index2+1,index2-1
def buildEdgeTable(puzzle1,puzzle2):
    '''
    in one D
    '''
    
    puzzle1_list=[x[0] for x in puzzle1]
    puzzle2_list=[x[0] for x in puzzle2]

    edge_table={}
    for i in range(Rowsize*Colsize):
        edge_table[str(puzzle1_list[i])]=[]
        if i == Rowsize*Colsize-1:
            edge_table[str(puzzle1_list[i])].append(puzzle1_list[0])
            edge_table[str(puzzle1_list[i])].append(puzzle1_list[i-1])
        else:
            edge_table[str(puzzle1_list[i])].append(puzzle1_list[i-1])
            edge_table[str(puzzle1_list[i])].append(puzzle1_list[i+1])
        puzzle2_edges_index1,puzzle2_edges_index2=find_puzzle2_edge(puzzle2_list,puzzle1_list[i])
        edge_table[str(puzzle1_list[i])].append(puzzle2_list[puzzle2_edges_index1])
        edge_table[str(puzzle1_list[i])].append(puzzle2_list[puzzle2_edges_index2])
    return edge_table,puzzle1_list,puzzle2_list
def find_puzzle2_2dedge(puzzle_list,element):
    index2=puzzle_list.index(element)
    puzzle2_row=index2//Rowsize
    puzzle2_col=index2%Rowsize
    puzzle2_edge_result=[]
    if puzzle2_col == Rowsize-1:
        puzzle2_edge_result.append([puzzle2_row,0])
        puzzle2_edge_result.append([puzzle2_row,puzzle2_col-1])
    else:
        puzzle2_edge_result.append([puzzle2_row,puzzle2_col+1])
        puzzle2_edge_result.append([puzzle2_row,puzzle2_col-1])
    if puzzle2_row == Colsize-1:
        puzzle2_edge_result.append([0,puzzle2_col])
        puzzle2_edge_result.append([puzzle2_row-1,puzzle2_col])
    else:
        puzzle2_edge_result.append([puzzle2_row+1,puzzle2_col])
        puzzle2_edge_result.append([puzzle2_row-1,puzzle2_col])
    return puzzle2_edge_result
def build2DEdgeTable(puzzle1,puzzle2):
    '''
    in 2D
    '''
    puzzle1_2D=reshape(puzzle1,Rowsize,Colsize)
    puzzle2_2D=reshape(puzzle2,Rowsize,Colsize)
    puzzle1_list=[x[0] for x in puzzle1]
    puzzle2_list=[x[0] for x in puzzle2]
    edge2d_table={}
    for i in range(Rowsize*Colsize):
        edge2d_table[str(puzzle1_list[i])]=[]
        puzzle1_row=i//Rowsize
        puzzle1_col=i%Rowsize
        if puzzle1_col == Rowsize-1:
            edge2d_table[str(puzzle1_list[i])].append(puzzle1_2D[puzzle1_row][0][0])
            edge2d_table[str(puzzle1_list[i])].append(puzzle1_2D[puzzle1_row][puzzle1_col-1][0])
        else:
            edge2d_table[str(puzzle1_list[i])].append(puzzle1_2D[puzzle1_row][puzzle1_col+1][0])
            edge2d_table[str(puzzle1_list[i])].append(puzzle1_2D[puzzle1_row][puzzle1_col-1][0])
        if puzzle1_row == Colsize-1:
            edge2d_table[str(puzzle1_list[i])].append(puzzle1_2D[0][puzzle1_col][0])
            edge2d_table[str(puzzle1_list[i])].append(puzzle1_2D[puzzle1_row-1][puzzle1_col][0])
        else:
            edge2d_table[str(puzzle1_list[i])].append(puzzle1_2D[puzzle1_row+1][puzzle1_col][0])
            edge2d_table[str(puzzle1_list[i])].append(puzzle1_2D[puzzle1_row-1][puzzle1_col][0])
            
        puzzle2_edge_result=find_puzzle2_2dedge(puzzle2_list,puzzle1_list[i])
        for x,y in puzzle2_edge_result:
            edge2d_table[str(puzzle1_list[i])].append(puzzle2_2D[x][y][0])
    return edge2d_table,puzzle1_list,puzzle2_list
def update_edge_table(edge_table,element):
    for key in edge_table:
        if element in edge_table[key]:
            edge_table[key]=[x for x in edge_table[key] if x != element]
    return edge_table
def find_min_or_random(lst): # 找最短的或随机选
    min_value = min(lst)
    min_values = [x for x in lst if x == min_value]
    if len(min_values) == 1:
        return min_value
    else:
        return random.choice(min_values)
def select_perfect_element(edge_table,element):
    # print('last select',element)
    
    edges_list=edge_table[str(element)]
    edge_table={key: value for key, value in edge_table.items() if key!=str(element)} #删除用完的key
    if not edges_list:
        
        element=random.choice(list(edge_table.items()))[0]
        # print('random select',element)
        perfect_element=int(element)
        edge_table=update_edge_table(edge_table,perfect_element)
        return perfect_element,edge_table
    # print(element,edges_list)

    # if len(edges_list)==0:
    #     perfect_element=int(element)
    #     edge_table=update_edge_table(edge_table,perfect_element)
    #     return perfect_element,edge_table
    edge_length_list=[len(edge_table[str(x)]) for x in edges_list]
    # print(edges_list,edge_length_list)
    temp_list=[]
    for edge in edges_list:
        if edge not in temp_list:
            temp_list.append(edge)
        else:
            perfect_element=edge
            edge_table=update_edge_table(edge_table,perfect_element)
            return perfect_element,edge_table
    select_index=edge_length_list.index(find_min_or_random(edge_length_list))
    perfect_element=edges_list[select_index]
    edge_table=update_edge_table(edge_table,perfect_element)
    return perfect_element,edge_table

def EdgeRecombination(puzzle1,puzzle2):
    puzzle1_code=encode_dictionary(puzzle1)
    puzzle2_code=encode_dictionary(puzzle2)
    edge_table,puzzle1_list,puzzle2_list=buildEdgeTable(puzzle1,puzzle2)
    random_num=random.randint(0,Rowsize*Colsize-1)
    select_element1=puzzle1_list[random_num]
    select_element2=puzzle2_list[random_num]

    edge_table1=copy.deepcopy(edge_table)
    edge_table2=copy.deepcopy(edge_table)
    edge_table1=update_edge_table(edge_table1,select_element1)
    edge_table2=update_edge_table(edge_table2,select_element2)
    c1,c2=[],[]
    for i in range(1,Rowsize*Colsize):
        c1.append(select_element1)
        
        c2.append(select_element2)
        # print('c1,c2',c1,c2)
        # print('C1')
        select_element1,edge_table1=select_perfect_element(edge_table1,select_element1)
        # print('C2')
        select_element2,edge_table2=select_perfect_element(edge_table2,select_element2)
    c1.append(select_element1)
    c2.append(select_element2)
    child1=[[x,puzzle1_code[str(x)]]for x in c1]
    child2=[[x,puzzle2_code[str(x)]]for x in c2]
    return child1,child2

def EdgeRecombination2D(puzzle1,puzzle2):
    puzzle1_code=encode_dictionary(puzzle1)
    puzzle2_code=encode_dictionary(puzzle2)
    edge2d_table,puzzle1_list,puzzle2_list=build2DEdgeTable(puzzle1,puzzle2)
    
    random_num=random.randint(0,Rowsize*Colsize-1)
    select_element1=puzzle1_list[random_num]
    select_element2=puzzle2_list[random_num]

    edge_table1=copy.deepcopy(edge2d_table)
    edge_table2=copy.deepcopy(edge2d_table)
    edge_table1=update_edge_table(edge_table1,select_element1)
    edge_table2=update_edge_table(edge_table2,select_element2)
    c1,c2=[[0 for _ in range(Colsize)] for _ in range(Rowsize)],[[0 for _ in range(Colsize)] for _ in range(Rowsize)]
    for i in range(Rowsize):
        for j in range(Colsize):
            if i%2==0:
                c1[i][j]=select_element1
                c2[i][j]=select_element2
            else:
                c1[i][-j-1]=select_element1
                c2[i][-j-1]=select_element2
            # print('c1,c2',c1,c2)
            # print('C1')
            if i==Rowsize-1 and j==Colsize-1:
                break
            select_element1,edge_table1=select_perfect_element(edge_table1,select_element1)
            # print('C2')
            select_element2,edge_table2=select_perfect_element(edge_table2,select_element2)
    c1=flatten(c1)
    c2=flatten(c2)
    child1=[[x,puzzle1_code[str(x)]]for x in c1]
    child2=[[x,puzzle2_code[str(x)]]for x in c2]
    return child1,child2
def write_file(best_solution):
    best_solution=reshape(best_solution,Rowsize,Colsize)
    best_solution=[[x[1] for x in row] for row in best_solution]
    file_name = "Ass1Output_edge.txt"
    with open(file_name, 'w') as f:
        f.write(f"yuhangchen,Jiaxiyang\n")
        for row in best_solution:
            f.write(' '.join(row) + '\n')


def localSearch(puzzle):
    best_fitness = calculateFitness(puzzle)
    best_puzzle = puzzle.copy()
    for _ in range(100):  # 迭代次数
        # 生成邻域解
        neighbor = mutation1(puzzle.copy())
        fitness = calculateFitness(neighbor)
        if fitness < best_fitness:
            best_fitness = fitness
            best_puzzle = neighbor.copy()
    return best_puzzle

def main():
    global swap_probability,rotate_probability
    population=initionation()
    fitness_board=list(map(calculateFitness,population))
    best_fitness=min(fitness_board)
    Generation=0
    best_individual=population[fitness_board.index(best_fitness)]
    fitness_list=[]
    while fitness_board[fitness_board.index(min(fitness_board))]>0 and Generation<maxGeneration:
        # print(swap_probability,rotate_probability)
        random_parent=random.sample(range(0, population_size), int(population_size*offspringPercent)) #随机选择offspringPercent的父代
        windows=[[i,population[i]] for i in random_parent]
        windows.append([fitness_board.index(best_fitness),best_individual])
        windows.sort(key=lambda x:calculateFitness(x[1]))
        new_population=[]
        for i in range(len(windows)//2):
            parent1=windows[i*2][1]
            parent2=windows[i*2+1][1]
            # child1,child2=Crossover(parent1,parent2,3,3)
            # child1,child2=EdgeRecombination(parent1,parent2)
            child1,child2=EdgeRecombination2D(parent1,parent2)
            child1=mutation1(child1)
            child2=mutation1(child2)
            # child1=localSearch(child1)
            # child2=localSearch(child2)
            child1=mutation2(child1)
            child2=mutation2(child2)
            new_population.append(child1)
            new_population.append(child2)
        # windows+=new_population
        # windows.sort(key=lambda x:calculateFitness(x))
        # print(len(windows),len(new_population))
        new_population.sort(key=lambda x:calculateFitness(x))
        for offspring in new_population:
            for index,old_population in enumerate(windows):
                if calculateFitness(offspring)<calculateFitness(old_population[1]):
                    windows[index][1]=offspring
                    break
        for new_offspring in windows:
            population[new_offspring[0]]=new_offspring[1]
        fitness_board=list(map(calculateFitness,population))
        best_fitness=min(fitness_board)
        fitness_list.append(best_fitness)
        Generation+=1
        print(best_fitness)
        best_individual=population[fitness_board.index(best_fitness)]
        best_individual=localSearch(best_individual)
        population[fitness_board.index(best_fitness)]=best_individual
        if Generation % 3 ==0:
            if best_fitness==fitness_list[-3]:
                swap_probability+=(100-swap_probability)*0.5
                rotate_probability+=(100-rotate_probability)*0.5
            else:
                swap_probability*=0.85
                rotate_probability*=0.85
            # population.sort(key=lambda x:calculateFitness(x))
            # num_replace = int(0.5 * population_size)
            # new_individuals = initionation()[:num_replace]
            # population[-num_replace:] = new_individuals
    best_solution=population[fitness_board.index(best_fitness)]
    write_file(best_solution)
if __name__ == '__main__':
    main()