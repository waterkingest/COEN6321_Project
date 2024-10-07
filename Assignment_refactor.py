import random
import numpy as np
import copy
from scipy.optimize import linear_sum_assignment
import os
def Creat_code_dictionary():
    '''
    Read raw puzzle file to creat Genotype dictionary
    '''
    input_puzzle=[]
    with open(input_file_path, 'r') as f:
        for line in f:
            input_puzzle.append(line.strip().split(' '))
    Code_dictionary={}
    code=1
    for i in range(1,Rowsize+1):
        for j in range(1,Colsize+1):
            Code_dictionary[str(code)]=input_puzzle[i-1][j-1]
            code+=1
    return Code_dictionary
def decode_dictionary(piece):
    '''
    Decode Genotype to puzzle
    '''
    global Code_dictionary
    return Code_dictionary[piece]
def initialization():
    '''
    Initialize population
    Our genotype of each solution like 
    [[piece_id_1, piece_1, piece_1 angle],[piece_id_2, piece_2, piece_2 angle],...]
    A list of 1*64
    '''
    population=[]
    for i in range(population_size):
        Code_puzzle=random.sample(range(1, Rowsize*Colsize+1), Rowsize*Colsize)
        puzzle=[[x,decode_dictionary(str(x)),0]for x in Code_puzzle]
        population.append(puzzle)
    return population
def calculatRowMisMatch(puzzleRow1,puzzleRow2,r1,r2):
    '''
    Calculate the mismatch between two rows,
    used for calculate the fitness
    '''
    global OutOrIN
    numberOfMisMatch=0
    for n,piece in enumerate(puzzleRow1):
        if piece[2]!=puzzleRow2[n][0]:
            if r1==0 or r2==Rowsize-1: 
                numberOfMisMatch+=OutOrIN # Variable penalization of edge area
            elif n==0 or n==Rowsize-1:
                numberOfMisMatch+=OutOrIN # Variable penalization of edge area
            else:
                numberOfMisMatch+=2
    return numberOfMisMatch

def calculatColMisMatch(puzzleCol1,puzzleCol2,c1,c2):
    '''
    Calculate the mismatch between two columns,
    used for calculate the fitness
    '''
    global OutOrIN
    numberOfMisMatch=0
    for n,piece in enumerate(puzzleCol1):
        if piece[1]!=puzzleCol2[n][3]:
            if c1==0 or c2==Colsize-1: 
                numberOfMisMatch+=OutOrIN # Variable penalization of edge area
            elif n==0 or n==Colsize-1: 
                numberOfMisMatch+=OutOrIN # Variable penalization of edge area
            else:
                numberOfMisMatch+=2
    return numberOfMisMatch

def calculatRowMisMatch2(puzzleRow1,puzzleRow2):
    '''
    Calculate the mismatch between two rows,
    used for calculate the mismatch values
    '''
    numberOfMisMatch=0
    for n,piece in enumerate(puzzleRow1):
        if piece[2]!=puzzleRow2[n][0]:
            numberOfMisMatch+=1
    return numberOfMisMatch

def calculatColMisMatch2(puzzleCol1,puzzleCol2):
    '''
    Calculate the mismatch between two columns,
    used for calculate the mismatch values
    '''
    numberOfMisMatch=0
    for n,piece in enumerate(puzzleCol1):
        if piece[1]!=puzzleCol2[n][3]:
            numberOfMisMatch+=1
    return numberOfMisMatch

def calculateFitness(puzzle):
    '''
    Calculate the fitness of a puzzle
    '''
    puzzle=[x[1] for x in puzzle]
    puzzle=np.array(puzzle).reshape(Rowsize, Colsize).tolist()
    fitness=0
    for n in range(Rowsize-1):
        fitness+=calculatRowMisMatch(puzzle[n],puzzle[n+1],n,n+1)
    for n in range(Colsize-1):
        col1,col2=[],[]
        for i in range(Rowsize):
            col1.append(puzzle[i][n])
            col2.append(puzzle[i][n+1])
        fitness+=calculatColMisMatch(col1,col2,n,n+1)
    return fitness

def calculateMissmatch(puzzle):
    '''
    Calculate the missmatch of a puzzle
    '''
    puzzle=[x[1] for x in puzzle]
    puzzle=np.array(puzzle).reshape(Rowsize, Colsize).tolist()
    Missmatch=0
    for n in range(Rowsize-1):
        Missmatch+=calculatRowMisMatch2(puzzle[n],puzzle[n+1])
    for n in range(Colsize-1):
        col1,col2=[],[]
        for i in range(Rowsize):
            col1.append(puzzle[i][n])
            col2.append(puzzle[i][n+1])
        Missmatch+=calculatColMisMatch2(col1,col2)
    return Missmatch
def reshape(matrix,row,col):
    '''
    Reshape our solution to a 2D form (8*8)
    '''
    total_elements = row*col
    if total_elements != len(matrix):
        raise ValueError("Size Error")
    result=[]
    result=[matrix[i:i+col] for i in range(0, len(matrix), col)]
    return result

def flatten(matrix):
    '''
    Reshape our solution to a 1D form (1*64)
    '''
    flattened_arr = [item for submatrix in matrix for item in submatrix]
    return flattened_arr

def self_adaptive_Pm(current_value,min_value,max_value,improvement_rate,threshold_high=0.01,threshold_low=0.001,decay_factor=0.9,growth_factor=1.1):
    '''
    self-daptive probability of mutation
    '''
    if improvement_rate>threshold_high: # if improvement_rate is high, decrease the mutation rate
        new_value=max(current_value*decay_factor, min_value)
    elif improvement_rate<threshold_low:   #if improvement_rate is low, increase the mutation rate
        new_value=min(current_value*growth_factor, max_value)
    else:
        new_value = current_value
    return new_value
def mutation1(puzzle, mutation_rate, sigma):
    '''
    Exchange by block
    Size 1x1,1x2,2x1,2x2.
    '''
    if random.random()<mutation_rate:
        swap_numb=int(sigma)
        puzzle_2d=reshape(puzzle, Rowsize, Colsize)
        used_positions=set()
        for _ in range(swap_numb):
            block_sizes=[(1,1), (1,2), (2,1), (2,2)]
            block_size=random.choice(block_sizes)
            rows_block, cols_block = block_size
            # Trying to find blocks to prevent overlap
            for _ in range(100):  # Number of attempts
                # The first piece
                row1=random.randint(0,Rowsize-rows_block)
                col1=random.randint(0,Colsize-cols_block)
                # Check if it has been used by a previous block
                positions1=[(row1+r,col1+c) for r in range(rows_block) for c in range(cols_block)]
                if any(pos in used_positions for pos in positions1):
                    continue
                #The second piece
                row2=random.randint(0,Rowsize-rows_block)
                col2=random.randint(0,Colsize-cols_block)
                positions2=[(row2+r,col2+c) for r in range(rows_block) for c in range(cols_block)]
                if any(pos in used_positions for pos in positions2):
                    continue
                # Overlap or not, exchange if not
                if set(positions1).isdisjoint(set(positions2)):
                    for r in range(rows_block):
                        for c in range(cols_block):
                            temp_piece=puzzle_2d[row1+r][col1+c]
                            puzzle_2d[row1+r][col1+c]=puzzle_2d[row2+r][col2+c]
                            puzzle_2d[row2+r][col2+c]=temp_piece
                    # Marks the block as being used
                    used_positions.update(positions1)
                    used_positions.update(positions2)
                    break
        puzzle=flatten(puzzle_2d)
    return puzzle

def mutation2(puzzle, mutation_rate, sigma):
    '''
    Rotate the puzzle block and the pieces rotate with it.
    '''
    if random.random()<mutation_rate:
        puzzle_2d=reshape(puzzle, Rowsize, Colsize)
        used_positions=set()
        num_blocks=int(sigma)
        for _ in range(num_blocks):
            block_size=random.randint(1, 3)
            rows_block=cols_block=block_size
            # Trying to find blocks to prevent overlap
            for _ in range(100):
                row=random.randint(0,Rowsize-rows_block)
                col=random.randint(0,Colsize-cols_block)
                positions=[(row+r,col+c) for r in range(rows_block) for c in range(cols_block)]
                if any(pos in used_positions for pos in positions):
                    continue
                rotate_times = random.randint(1,3)
                block=[puzzle_2d[row+r][col+c] for r in range(rows_block) for c in range(cols_block)]
                block_2d=[block[i*cols_block:(i+1)*cols_block] for i in range(rows_block)]
                for _ in range(rotate_times):
                    block_2d=[list(x) for x in zip(*block_2d[::-1])]
                rotated_block=[item for sublist in block_2d for item in sublist]
                idx=0
                for r in range(rows_block):
                    for c in range(cols_block):
                        piece=rotated_block[idx]
                        piece[2]=(piece[2]+rotate_times)%4
                        piece[1]=piece[1][-rotate_times:]+piece[1][:-rotate_times]
                        puzzle_2d[row+r][col+c] = piece
                        idx +=1
                # Marks the block as being used
                used_positions.update(positions)
                break
        puzzle=flatten(puzzle_2d)
    return puzzle

def Crossover(puzzle1,puzzle2,window_row,window_col):
    '''
    Order Crossover, exange by region
    '''

    parent1=reshape(puzzle1,Rowsize,Colsize)
    parent2=reshape(puzzle2,Rowsize,Colsize)
    # Randomly select the location of the extraction block
    rand_row_index=random.randint(0,Rowsize-window_row)
    rand_col_index=random.randint(0,Colsize-window_col)
    part1,part2=[],[] #used to record the pieces in the parent
    record_p1,record_p2=[],[] #used to record the pieces in the window, to avoid duplication
    for row in parent1[rand_row_index:rand_row_index+window_row]:
        part1+=row[rand_col_index:rand_col_index+window_col]
        record_p1+=[x[0] for x in row[rand_col_index:rand_col_index+window_col]]
    for row in parent2[rand_row_index:rand_row_index+window_row]:
        part2+=row[rand_col_index:rand_col_index+window_col]
        record_p2+=[x[0] for x in row[rand_col_index:rand_col_index+window_col]]
    # Initialize a blank puzzle to create the child
    c1=np.zeros((Rowsize,Colsize),int).tolist()
    c2=np.zeros((Rowsize,Colsize),int).tolist()
    star=0
    # Putting the blocks extracted from the parent directly into the children
    for row_i in range(rand_row_index,rand_row_index+window_row):
        c1[row_i][rand_col_index:rand_col_index+window_col]=part1[star:star+window_col]
        c2[row_i][rand_col_index:rand_col_index+window_col]=part2[star:star+window_col]
        star+=window_col
    c1=flatten(c1)
    c2=flatten(c2)
    c1_index,c2_index=0,0
    # Put the rest of the non-duplicated pieces from the parent into the child in order.
    for i in range(Rowsize*Colsize):
        if  c1_index<Rowsize*Colsize and c1[c1_index]==0:
            if puzzle1[i][0] not in record_p1: # If it's not being used, put it in to the child
                c1[c1_index]=puzzle1[i]
                c1_index+=1
        else:
            c1_index+=window_col
        if c2_index<Rowsize*Colsize and c2[c2_index]==0 :
            if puzzle2[i][0] not in record_p2: # If it's not being used, put it in to the child
                c2[c2_index]=puzzle2[i]
                c2_index+=1
        else:
            c2_index+=window_col
    return c1,c2

def encode_dictionary(puzzle):
    '''
    encode the puzzle into a dictionary
    '''
    codeDictionary={}
    for i in range(1,Rowsize*Colsize+1):
        codeDictionary[str(puzzle[i-1][0])]=[puzzle[i-1][1],puzzle[i-1][2]]
    return codeDictionary

def find_puzzle2_edge(puzzle_list,element):
    '''
    find the edge from the puzzle 2 in 1d edge table
    '''
    index2=puzzle_list.index(element)
    if index2 == Rowsize*Colsize-1:
        return 0,index2-1
    else:
        return index2+1,index2-1

def buildEdgeTable(puzzle1,puzzle2):
    '''
    Build the edge table for 21 Dimension, each piece has 2 edges
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
    '''
    find the edge from the puzzle 2 in 2d edge table
    '''
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
    Build the edge table for 2 Dimension, each piece has 4 edges
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
    '''
    Remove the selected element from the Edge table
    '''
    for key in edge_table:
        if element in edge_table[key]:
            edge_table[key]=[x for x in edge_table[key] if x != element]
    return edge_table
def find_min_or_random(lst):
    '''
    Find the shortest or random select from the Edge table
    '''
    min_value = min(lst)
    min_values = [x for x in lst if x == min_value]
    if len(min_values) == 1:
        return min_value
    else:
        return random.choice(min_values)
def select_perfect_element(edge_table,element):
    '''
    Select the perfect element from the edge table
    '''
    edges_list=edge_table[str(element)]
    edge_table={key: value for key, value in edge_table.items() if key!=str(element)} # Remove the selected element from the edge table
    if not edges_list: # If the edge list is empty, random select an element from the edge table
        
        element=random.choice(list(edge_table.items()))[0]
        perfect_element=int(element)
        edge_table=update_edge_table(edge_table,perfect_element) # Remove the selected element from the edge table
        return perfect_element,edge_table
    edge_length_list=[len(edge_table[str(x)]) for x in edges_list]
    temp_list=[]
    for edge in edges_list:
        if edge not in temp_list:
            temp_list.append(edge)
        else:
            perfect_element=edge # find the common edge
            edge_table=update_edge_table(edge_table,perfect_element) # Remove the selected element from the edge table
            return perfect_element,edge_table
    select_index=edge_length_list.index(find_min_or_random(edge_length_list)) # if cannot find the common edge, select the shortest edge or reandom select
    perfect_element=edges_list[select_index]
    edge_table=update_edge_table(edge_table,perfect_element) # Remove the selected element from the edge table
    return perfect_element,edge_table

def EdgeRecombination(puzzle1,puzzle2):
    '''
    Do edge recombination in one dimension
    '''
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
        select_element1,edge_table1=select_perfect_element(edge_table1,select_element1)
        select_element2,edge_table2=select_perfect_element(edge_table2,select_element2)
    c1.append(select_element1)
    c2.append(select_element2)
    child1=[[x,puzzle1_code[str(x)][0],puzzle1_code[str(x)][1]]for x in c1]
    child2=[[x,puzzle2_code[str(x)][0],puzzle2_code[str(x)][1]]for x in c2]
    return child1,child2

def EdgeRecombination2D(puzzle1,puzzle2):
    '''
    Do edge recombination in two dimension
    '''
    puzzle1_code=encode_dictionary(puzzle1)
    puzzle2_code=encode_dictionary(puzzle2)
    # Step1: creat the edge table
    edge2d_table,puzzle1_list,puzzle2_list=build2DEdgeTable(puzzle1,puzzle2)
    
    random_num=random.randint(0,Rowsize*Colsize-1)
    # Step2: random select the first element
    select_element1=puzzle1_list[random_num]
    select_element2=puzzle2_list[random_num]

    edge_table1=copy.deepcopy(edge2d_table)
    edge_table2=copy.deepcopy(edge2d_table)
    # remove the selected element from the edge table
    edge_table1=update_edge_table(edge_table1,select_element1)
    edge_table2=update_edge_table(edge_table2,select_element2)
    # initialize the child
    c1,c2=[[0 for _ in range(Colsize)] for _ in range(Rowsize)],[[0 for _ in range(Colsize)] for _ in range(Rowsize)]
    for i in range(Rowsize):
        for j in range(Colsize):
            # follow the S sequence to select the element
            if i%2==0:
                c1[i][j]=select_element1
                c2[i][j]=select_element2
            else:
                c1[i][-j-1]=select_element1
                c2[i][-j-1]=select_element2
            if i==Rowsize-1 and j==Colsize-1:
                break
            # Step3: select the next element
            select_element1,edge_table1=select_perfect_element(edge_table1,select_element1)
            select_element2,edge_table2=select_perfect_element(edge_table2,select_element2)
    c1=flatten(c1)
    c2=flatten(c2)
    child1=[[x,puzzle1_code[str(x)][0],puzzle1_code[str(x)][1]]for x in c1]
    child2=[[x,puzzle2_code[str(x)][0],puzzle2_code[str(x)][1]]for x in c2]
    return child1,child2
def write_file(best_solution,best_mismatch):
    '''
    write the best solution to the file
    '''
    best_solution=reshape(best_solution,Rowsize,Colsize)
    best_solution=[[x[1] for x in row] for row in best_solution]
    file_name = "{}_{}.txt".format(output_file_name,best_mismatch)
    with open(file_name, 'w') as f:
        f.write(f"Yuhang Chen 40253925,Jiaxi Yang 40261989\n")
        for row in best_solution:
            f.write(' '.join(row) + '\n')
def format_solution(best_solution):
    '''
    format the best solution and print it
    '''
    print('Best Result: ')
    best_solution=reshape(best_solution,Rowsize,Colsize)
    best_solution=[[x[1] for x in row] for row in best_solution]
    for row in best_solution:
        print('|-----|'*len(row))
        up_number_list=[x[0] for x in row]
        left_right_list=[x[1:4:2][::-1] for x in row]
        down_number_list=[x[2] for x in row]
        for up_num in up_number_list:
            print(f"|  {up_num}  |",end='')
        print('')
        for left_right_num in left_right_list:
            print(f"|{left_right_num[0]}   {left_right_num[1]}|",end='')
        print('')
        for down_num in down_number_list:
            print(f"|  {down_num}  |",end='')
        print('')
    print('|-----|'*len(row))
    return best_solution

def build_distance_matrix(population):
    '''
    build the distance matrix to record the distance between each individual
    '''
    dis_value=np.zeros((population_size, population_size)).tolist()
    for i in range(len(population)):
        for j in range(i,len(population)):
            if i ==j:
                dis_value[i][j]=[999,j]
            else:
                dis_value[i][j]=[cal_distance(population[i],population[j]),j]
    for i in range(len(population)):
        for j in range(i):
            dis_value[i][j] = dis_value[j][i]
    return dis_value
def cal_distance(puzzle1,puzzle2):
    '''
    calculate the distance between two puzzles
    '''
    for idv_tile in range(Colsize*Rowsize):
        angle=abs((puzzle1[idv_tile][2]-puzzle2[idv_tile][2])%3+(puzzle1[idv_tile][2]-puzzle2[idv_tile][2])//3) #calculate the angle difference
        id_diff=puzzle1[idv_tile][0]-puzzle2[idv_tile][0]
        if id_diff !=0:
            distance=8     #set the weight
        else:
            distance=angle
    return distance

#VLNS
def select_non_adjacent_positions(k):
    positions = []
    attempts = 0
    max_attempts = 1000
    while len(positions) < k and attempts < max_attempts:
        row = random.randint(0, Rowsize - 1)
        col = random.randint(0, Colsize - 1)
        pos = (row, col)
        # Check for proximity to the selected position
        adjacent = False
        for p in positions:
            if abs(p[0] - row) + abs(p[1] - col) == 1:
                adjacent = True
                break
        if not adjacent and pos not in positions:
            positions.append(pos)
        attempts += 1
    return positions


def rotate_piece(piece, r):
    '''
    Rotate a piece by r degrees (0, 90, 180, or 270)
    '''
    id, edges, angle = piece
    new_angle = (angle + r) % 4
    new_edges = edges[-r:] + edges[:-r]
    return [id, new_edges, new_angle]

def compute_matching_edges(puzzle, idx):
    '''
    Calculate the number of mismatching sides for a given puzzle piece placement
    '''
    global OutOrIN
    total_mismatch = 0
    row = idx // Colsize
    col = idx % Colsize
    piece = puzzle[idx]
    piece_edges = piece[1]
    # Check the neighbors in each direction
    # Up neighbor
    if row > 0:
        neighbor_idx = (row - 1) * Colsize + col
        neighbor_piece = puzzle[neighbor_idx]
        if neighbor_piece:
            neighbor_edge = neighbor_piece[1][2]
            if piece_edges[0] != neighbor_edge:
                total_mismatch += 1
    else:
        total_mismatch += OutOrIN  # Edge Penalty
    # right neighbor
    if col < Colsize - 1:
        neighbor_idx = row * Colsize + (col + 1)
        neighbor_piece = puzzle[neighbor_idx]
        if neighbor_piece:
            neighbor_edge = neighbor_piece[1][3]
            if piece_edges[1] != neighbor_edge:
                total_mismatch += 1
    else:
        total_mismatch += OutOrIN # Edge Penalty
    # down neighbor
    if row < Rowsize - 1:
        neighbor_idx = (row + 1) * Colsize + col
        neighbor_piece = puzzle[neighbor_idx]
        if neighbor_piece:
            neighbor_edge = neighbor_piece[1][0]
            if piece_edges[2] != neighbor_edge:
                total_mismatch += 1
    else:
        total_mismatch += OutOrIN # Edge Penalty
    # left neighbor
    if col > 0:
        neighbor_idx = row * Colsize + (col - 1)
        neighbor_piece = puzzle[neighbor_idx]
        if neighbor_piece:
            neighbor_edge = neighbor_piece[1][1]
            if piece_edges[3] != neighbor_edge:
                total_mismatch += 1
    else:
        total_mismatch += OutOrIN # Edge Penalty
    return total_mismatch

def VLNS(puzzle, k):
    '''
    Vary large neighborhood search algorithm
    '''
    # Step 1: Select k non-adjacent positions
    positions = select_non_adjacent_positions(k)
    positions_indices = [row * Colsize + col for (row, col) in positions]

    # Step 2: Remove the pieces at the selected positions
    removed_pieces = [puzzle[idx] for idx in positions_indices]
    
    # Step 3: Mark Remove Puzzle as Empty
    temp_puzzle = puzzle.copy()
    for idx in positions_indices:
        temp_puzzle[idx] = None  # 标记为空

    n = len(removed_pieces)
    w_matrix = np.zeros((n, n))
    r_matrix = np.zeros((n, n), dtype=int)

    # Step 4: Compute the mismatching edges for each pair of removed pieces
    for i, piece in enumerate(removed_pieces):
        for j, hole_pos in enumerate(positions):
            min_w = float('inf')
            best_r = 0
            for r in range(4):
                rotated_piece = rotate_piece(piece, r)
                idx = hole_pos[0] * Colsize + hole_pos[1]
                temp_puzzle_copy = temp_puzzle.copy()
                temp_puzzle_copy[idx] = rotated_piece
                w = compute_matching_edges(temp_puzzle_copy, idx)
                if w < min_w:
                    min_w = w
                    best_r = r
            w_matrix[i][j] = min_w  # Record the lowest number of mismatches
            r_matrix[i][j] = best_r

    # Using the Hungarian algorithm to find the best match
    row_ind, col_ind = linear_sum_assignment(w_matrix)

    # Step 5: Place the pieces back into the puzzle
    for i, j in zip(row_ind, col_ind):
        piece = removed_pieces[i]
        rotate_times = r_matrix[i][j]
        rotated_piece = rotate_piece(piece, rotate_times)
        hole_pos = positions[j]
        idx = hole_pos[0] * Colsize + hole_pos[1]
        temp_puzzle[idx] = rotated_piece

    return temp_puzzle


def localSearch_VLNS(puzzle):
    '''
    Vary large neighborhood search
    '''
    best_puzzle = VLNS(puzzle, VLNS_Size)
    best_fitness = calculateFitness(best_puzzle)
    original_fitness = calculateFitness(puzzle)
    if best_fitness < original_fitness:
        return best_puzzle
    else:
        return puzzle

def extra_population(population_X):
    '''
    Renew the worst population
    '''
    new_population=[]
    sigma=Rowsize*Colsize*0.5
    for _ in range(len(population_X)//2):
        random_parent=random.sample(range(0, len(population_X)), Window_Size)
        windows=[[i,population_X[i]] for i in random_parent]
        windows.sort(key=lambda x:calculateFitness(x[1]))
        parent1=windows[0][1]
        parent2=windows[1][1]
        random_select=random.randint(0,100)
        if random_select<10:
            child1,child2=Crossover(parent1,parent2,3,3)
        elif random_select<40:
            child1,child2=EdgeRecombination(parent1,parent2)
        elif random_select<90:
            child1,child2=EdgeRecombination2D(parent1,parent2)
        else:
            child1,child2=parent1,parent2
        child1=mutation1(child1,1,sigma)
        child2=mutation1(child2,1,sigma)
        child1=mutation2(child1,1,sigma)
        child2=mutation2(child2,1,sigma)
        
        child1 = localSearch_VLNS(child1)
        child2 = localSearch_VLNS(child2)
        new_population.append(child1)
        new_population.append(child2)
    new_population+=population_X
    return new_population[:len(population_X)]

def main():
    global OutOrIN
    OutOrIN=1
    dict_fitness_value={"4":1,"1":4}# Used to record the variable penalization of edge area
    print("Initializing population...")
    population=initialization()
    fitness_board=list(map(calculateFitness,population))
    best_fitness=min(fitness_board)
    previous_best_fitness=best_fitness
    mutation_rate=initial_mutation_rate
    sigma=initial_sigma
    Generation=0
    fitness_list=[]
    real_best_fitness=999
    print('Starting evolution...')
    while fitness_board[fitness_board.index(min(fitness_board))]>0 and Generation<maxGeneration:
        distance_matrix=build_distance_matrix(population)
        '''
        Tournament Selection
        '''
        new_population=[]
        for _ in range(int(population_size*children_Percent)//2):
            random_parent=random.sample(range(0, population_size), Window_Size)
            windows=[[i,population[i]] for i in random_parent]
            windows.sort(key=lambda x:calculateFitness(x[1]))
            parent1=windows[0][1]  # choose the best parent follow the tournament selection method
            parent2=windows[1][1]
            xover_row=random.randint(1,Rowsize-1)
            xover_col=random.randint(1,Colsize-1)
            random_select=random.randint(0,100) # random select the crossover method
            if random_select<10:
                child1,child2=Crossover(parent1,parent2,xover_row,xover_col)
            elif random_select<40:
                child1,child2=EdgeRecombination(parent1,parent2)
            elif random_select<90:
                child1,child2=EdgeRecombination2D(parent1,parent2)
            else:
                child1,child2=parent1,parent2 # 10% chance of not crossover.
            child1=mutation1(child1,mutation_rate,sigma)
            child2=mutation1(child2,mutation_rate,sigma)
            child1=mutation2(child1,mutation_rate,sigma)
            child2=mutation2(child2,mutation_rate,sigma)
            child1 = localSearch_VLNS(child1)
            child2 = localSearch_VLNS(child2)
            new_population.append(child1)
            new_population.append(child2)
            new_population.append(parent1)
            new_population.append(parent2)
        new_population.sort(key=lambda x:calculateFitness(x))
        for index,old_population in enumerate(windows): #Keep, better solution
            population[random_parent[index]]=copy.deepcopy(new_population[index])
        #-----------------------------------------------------------------------------------------------------------------
        # Elitism
        fitness_board=list(map(calculateFitness,copy.deepcopy(population)))
        best_fitness=min(fitness_board)
        best_individual=population[fitness_board.index(best_fitness)]
        mismatch_Board=list(map(calculateMissmatch,copy.deepcopy(population)))
        best_individual=localSearch_VLNS(best_individual)
        population[fitness_board.index(best_fitness)]=best_individual
        # Random select individuals to do local search
        random_local_search=random.sample(range(0, population_size), population_size//2)
        for R_index in random_local_search:
            R_individual=population[R_index]
            R_individual = localSearch_VLNS(R_individual)
            population[R_index]=R_individual
        fitness_board=list(map(calculateFitness,copy.deepcopy(population)))
        best_fitness=min(fitness_board)
        fitness_list.append(best_fitness)
        
        
        if previous_best_fitness != 0:    #self-adaptive mutayion
            improvement_rate = (previous_best_fitness - best_fitness) / previous_best_fitness
        else:
            improvement_rate = 0
        mutation_rate = self_adaptive_Pm(mutation_rate,final_mutation_rate,initial_mutation_rate,improvement_rate)#self-adaptive mutayion
        sigma = self_adaptive_Pm(sigma,final_sigma,initial_sigma,improvement_rate)
        previous_best_fitness=best_fitness
        Generation+=1
        mismatch_Board=list(map(calculateMissmatch,population))
        best_solution=population[mismatch_Board.index(min(mismatch_Board))]
        best_mismatch=calculateMissmatch(best_solution)
        print(f'Generation {Generation}: -------------Best Mismatch = {best_mismatch}')
        if Generation % 5 ==0:
            if best_fitness==fitness_list[-3]:
                # if fitness did not improve for 3 times, update the penalty value to stimulate population to generate transformations
                OutOrIN=dict_fitness_value[str(OutOrIN)]
                num_replace = int(0.97 * population_size)
            else:
                # every 5 generations, update the worst 50% of population
                num_replace = int(0.5 * population_size)
            population.sort(key=lambda x:calculateFitness(x))
            
            new_individuals = extra_population(copy.deepcopy(population[-num_replace:])) #update the worst population
            population[-num_replace:] = copy.deepcopy(new_individuals)

        if best_fitness<real_best_fitness or best_mismatch<real_best_mismatch: # if generation fitness is better than the best fitness, update the best fitness, and write the file
            real_best_fitness=best_fitness
            real_best_mismatch=best_mismatch
            write_file(best_solution,best_mismatch)
    format_solution(best_solution)
    best_mismatch=calculateMissmatch(best_solution)
    write_file(best_solution,best_mismatch)
if __name__ == '__main__':
    # hyper parameter
    Rowsize=8
    Colsize=8
    children_Percent=0.4
    VLNS_Size=21
    Window_Size=5
    population_size=int(input("Enter the population size [100,1000]: "))
    maxGeneration=int(input('Enter the maxGeneration [1,100]: '))
    initial_mutation_rate=0.95
    final_mutation_rate=0.0001
    initial_sigma=Rowsize*Colsize*0.4
    final_sigma=1.0
    input_file_path=r'Example_Input&Output\Ass1Input.txt'
    output_file_name=r'Output\Ass1Output_Refactor_Add_VLNS'
    if not os.path.exists('Output'):
        os.makedirs('Output')
    # hyper parameter
    Code_dictionary=Creat_code_dictionary()
    main()