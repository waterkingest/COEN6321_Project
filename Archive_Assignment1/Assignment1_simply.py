# hyper parameter
import random
import os
import math
import numpy as np
import copy
Rowsize=8#8
Colsize=8#8
population_size=500
input_puzzle=[]
children_Percent=0.5
maxGeneration=500
initial_mutation_rate=0.95
final_mutation_rate=0.001
initial_sigma=Rowsize*Colsize*0.6
final_sigma=1.0
with open(r'Example_Input&Output\Ass1Input.txt', 'r') as f:
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

def initialization():
    population=[]
    for i in range(population_size):
        Code_puzzle = random.sample(range(1, Rowsize*Colsize+1), Rowsize*Colsize)
        puzzle = [[x,decode_dictionary(str(x)),0]for x in Code_puzzle]
        population.append(puzzle)
    return population


def calculatRowMisMatch(puzzleRow1,puzzleRow2,r1,r2):
    global OutOrIN
    numberOfMisMatch=0
    for n,piece in enumerate(puzzleRow1):
        if piece[2]!=puzzleRow2[n][0]:
            # if n==0 or n==Rowsize-1: #边缘的权值更小
            #     numberOfMisMatch+=1
            if r1==0 or r2==Rowsize-1: #边缘的权值更小
                numberOfMisMatch+=OutOrIN
            elif n==0 or n==Rowsize-1: #边缘的权值更小
                numberOfMisMatch+=OutOrIN
            else:
                numberOfMisMatch+=2
    return numberOfMisMatch
def calculatColMisMatch(puzzleCol1,puzzleCol2,c1,c2):
    global OutOrIN
    numberOfMisMatch=0
    for n,piece in enumerate(puzzleCol1):
        if piece[1]!=puzzleCol2[n][3]:
            # if n==0 or n==Colsize-1: #边缘的权值更小
            #     numberOfMisMatch+=1
            if c1==0 or c2==Colsize-1: #边缘的权值更小
                numberOfMisMatch+=OutOrIN
            elif n==0 or n==Colsize-1: #边缘的权值更小
                numberOfMisMatch+=OutOrIN
            else:
                numberOfMisMatch+=2
    return numberOfMisMatch
def calculatRowMisMatch2(puzzleRow1,puzzleRow2):
    numberOfMisMatch=0
    for n,piece in enumerate(puzzleRow1):
        if piece[2]!=puzzleRow2[n][0]:
            numberOfMisMatch+=1
    return numberOfMisMatch
def calculatColMisMatch2(puzzleCol1,puzzleCol2):
    numberOfMisMatch=0
    for n,piece in enumerate(puzzleCol1):
        if piece[1]!=puzzleCol2[n][3]:
            numberOfMisMatch+=1
    return numberOfMisMatch
def  calculateFitness(puzzle):
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
    # fitness=-fitness
    return fitness
def calculateMissmatch(puzzle):
    puzzle=[x[1] for x in puzzle]
    puzzle=np.array(puzzle).reshape(Rowsize, Colsize).tolist()
    fitness=0
    for n in range(Rowsize-1):
        fitness+=calculatRowMisMatch2(puzzle[n],puzzle[n+1])
    for n in range(Colsize-1):
        col1,col2=[],[]
        for i in range(Rowsize):
            col1.append(puzzle[i][n])
            col2.append(puzzle[i][n+1])
        fitness+=calculatColMisMatch2(col1,col2)
    # fitness=-fitness
    return fitness
def self_adaptive_Pm(current_value,min_value,max_value,improvement_rate,threshold_high=0.01,threshold_low=0.001,decay_factor=0.9,growth_factor=1.1):#self-daptive probability of mutation
    if improvement_rate>threshold_high:
        new_value=max(current_value*decay_factor, min_value)
    elif improvement_rate<threshold_low:
        new_value=min(current_value*growth_factor, max_value)
    else:
        new_value = current_value
    return new_value
def mutation11(puzzle, mutation_rate, sigma):
    '''
    Swap
    '''
    if random.random()<mutation_rate:
        swap_numb=random.randint(1, int(sigma))
        for _ in range(swap_numb):
            index1,index2=random.sample(range(Rowsize*Colsize), 2)
            puzzle[index1], puzzle[index2]=puzzle[index2], puzzle[index1]
    return puzzle
def mutation22(puzzle, mutation_rate, sigma):
    '''
    Rotate pieces with adaptive mutation rate and strength
    '''

    if random.random()<mutation_rate:
        rotation_numb=random.randint(1, int(sigma))
        for _ in range(rotation_numb):
            rotate_times=random.randint(1, 3)
            index=random.randint(0, Rowsize * Colsize - 1)
            puzzle[index][2] = (puzzle[index][2] + rotate_times) % 4
            puzzle[index][1]=puzzle[index][1][-rotate_times:]+puzzle[index][1][:-rotate_times]
    return puzzle
def mutation1(puzzle, mutation_rate, sigma):
    '''
    按块交换
    尺寸有1x1,1x2,2x1,2x2.
    '''
    if random.random()<mutation_rate:
        # swap_numb=random.randint(1,int(sigma))
        swap_numb=int(sigma)    #这个我不确定自适应要1到sigma取整的随机数还是从sigma自适应下降
        puzzle_2d=reshape(puzzle, Rowsize, Colsize)
        used_positions=set()
        for _ in range(swap_numb):
            block_sizes=[(1,1), (1,2), (2,1), (2,2)]
            block_size=random.choice(block_sizes)
            rows_block, cols_block = block_size
            # 尝试找块数防止重叠
            success = False
            for attempt in range(66):  #尝试子次数
                #第一个块
                row1=random.randint(0,Rowsize-rows_block)
                col1=random.randint(0,Colsize-cols_block)
                #是否被之前的块用过
                positions1=[(row1+r,col1+c) for r in range(rows_block) for c in range(cols_block)]
                if any(pos in used_positions for pos in positions1):
                    continue
                #第二个快
                row2=random.randint(0,Rowsize-rows_block)
                col2=random.randint(0,Colsize-cols_block)
                positions2=[(row2+r,col2+c) for r in range(rows_block) for c in range(cols_block)]
                if any(pos in used_positions for pos in positions2):
                    continue
                #是否重叠，不重叠就交换
                if set(positions1).isdisjoint(set(positions2)):
                    for r in range(rows_block):
                        for c in range(cols_block):
                            temp_piece=puzzle_2d[row1+r][col1+c]
                            puzzle_2d[row1+r][col1+c]=puzzle_2d[row2+r][col2+c]
                            puzzle_2d[row2+r][col2+c]=temp_piece
                    #标记该块被用
                    used_positions.update(positions1)
                    used_positions.update(positions2)
                    success=True
                    break
            if not success:
                pass
        puzzle=flatten(puzzle_2d)
    return puzzle
# def mutation2(puzzle, mutation_rate, sigma):
#     '''
#     按块旋转,块为1x1大小时,旋转拼图块本身；
#     块为2x2及以上大尺寸时，仅旋转块中拼图块的位置，不改变其内部序列。
#     '''
#     if random.random() < mutation_rate:
#         puzzle_2d=reshape(puzzle, Rowsize, Colsize)
#         #记录已使用的位置
#         used_positions=set()
#         num_blocks=random.randint(1, int(sigma))
#         # num_blocks=int(sigma)    #这个我不确定自适应要1到sigma取整的随机数还是从sigma自适应下降
#         for _ in range(num_blocks):
#             block_size=random.randint(1,3)
#             rows_block=cols_block = block_size
#             success=False
#             for attempt in range(66):  # 尝试次数（保证块不能出界）
#                 row=random.randint(0,Rowsize-rows_block)
#                 col=random.randint(0,Colsize-cols_block)
#                 # 获取块的坐标
#                 positions=[(row+r,col+c) for r in range(rows_block) for c in range(cols_block)]
#                 # 检查位置是否被使用
#                 if any(pos in used_positions for pos in positions):
#                     continue  # 如重叠就重选
#                 rotate_times=random.randint(1, 3)
#                 block=[puzzle_2d[row+r][col+c] for r in range(rows_block) for c in range(cols_block)]
#                 block_2d=[block[i*cols_block:(i+1)*cols_block] for i in range(rows_block)]
#                 for _ in range(rotate_times):
#                     block_2d=[list(x) for x in zip(*block_2d[::-1])]
#                 if block_size==1:#如果是1x1就对内容进行旋转
#                     rotated_block=[item for sublist in block_2d for item in sublist]
#                     piece=rotated_block[0]
#                     rotate_times_piece=rotate_times%4
#                     piece[2]=(piece[2]+rotate_times_piece)%4
#                     piece[1]=piece[1][-rotate_times_piece:]+piece[1][:-rotate_times_piece]
#                     puzzle_2d[row][col]=piece
#                 else:
#                     # 块尺寸大于1x1只旋转拼图块，不改变其内部上右下左所对应的值
#                     rotated_block=[item for sublist in block_2d for item in sublist]
#                     idx=0  #遍历旋转后的块
#                     for r in range(rows_block):
#                         for c in range(cols_block):
#                             piece=rotated_block[idx]
#                             puzzle_2d[row+r][col+c] = piece
#                             idx += 1
#                 # 标记已占用位置，之后的重叠
#                 used_positions.update(positions)
#                 success=True
#                 break
#             if not success:
#                 pass
#         puzzle=flatten(puzzle_2d)
#     puzzle=localSearch(puzzle)
#     return puzzle

def mutation2(puzzle, mutation_rate, sigma):
    '''
    按块旋转版本2 ，tile里的顺序也会随着外面旋转而旋转
    '''
    if random.random()<mutation_rate:
        puzzle_2d=reshape(puzzle, Rowsize, Colsize)
        used_positions=set()
        # num_blocks=random.randint(1,int(sigma))
        num_blocks=int(sigma)
        for _ in range(num_blocks):
            block_size=random.randint(1, 3)
            rows_block=cols_block=block_size
            #找不重叠的
            success = False
            for attempt in range(66):
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
                #标记被用的区域
                used_positions.update(positions)
                success=True
                break
            if not success:
                pass
        puzzle=flatten(puzzle_2d)
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
        codeDictionary[str(puzzle[i-1][0])]=[puzzle[i-1][1],puzzle[i-1][2]]
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
    edge_length_list=[len(edge_table[str(x)]) for x in edges_list]
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
        select_element1,edge_table1=select_perfect_element(edge_table1,select_element1)
        select_element2,edge_table2=select_perfect_element(edge_table2,select_element2)
    c1.append(select_element1)
    c2.append(select_element2)
    child1=[[x,puzzle1_code[str(x)][0],puzzle1_code[str(x)][1]]for x in c1]
    child2=[[x,puzzle2_code[str(x)][0],puzzle2_code[str(x)][1]]for x in c2]
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
            if i==Rowsize-1 and j==Colsize-1:
                break
            select_element1,edge_table1=select_perfect_element(edge_table1,select_element1)
            select_element2,edge_table2=select_perfect_element(edge_table2,select_element2)
    c1=flatten(c1)
    c2=flatten(c2)
    child1=[[x,puzzle1_code[str(x)][0],puzzle1_code[str(x)][1]]for x in c1]
    child2=[[x,puzzle2_code[str(x)][0],puzzle2_code[str(x)][1]]for x in c2]
    return child1,child2
def write_file(best_solution,best_mismatch):
    best_solution=reshape(best_solution,Rowsize,Colsize)
    best_solution=[[x[1] for x in row] for row in best_solution]
    file_name = r"Output/Ass1Output_Simply_{}.txt".format(best_mismatch)
    with open(file_name, 'w') as f:
        f.write(f"yuhangchen,Jiaxiyang\n")
        for row in best_solution:
            f.write(' '.join(row) + '\n')
def format_solution(best_solution):
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

def localSearch(puzzle,mutation_rate=0.9, sigma=1):
    tabu_list = []
    best_fitness = calculateFitness(puzzle)
    # print('input mutaition1:',best_fitness)
    best_puzzle = copy.deepcopy(puzzle)
    for _ in range(100):  # 迭代次数
        # 生成邻域解
        # mutation_rate=0.9
        # sigma=1.0
        neighbor = mutation11(copy.deepcopy(best_puzzle),mutation_rate, sigma)
        fitness = calculateFitness(neighbor)
        if fitness < best_fitness and neighbor not in tabu_list:
            best_fitness = fitness
            best_puzzle = copy.deepcopy(neighbor)
        tabu_list.append(neighbor)
    # print('output mutation1',calculateFitness(best_puzzle))
    return best_puzzle
def localSearch2(puzzle,mutation_rate=0.9, sigma=1):
    tabu_list = []
    best_fitness = calculateFitness(puzzle)
    # print('input mutaition2:',best_fitness)
    best_puzzle =copy.deepcopy(puzzle)
    for _ in range(100):  # 迭代次数
        # 生成邻域解
        # mutation_rate=0.9
        # sigma=1.0
        neighbor = mutation22(copy.deepcopy(best_puzzle),mutation_rate, sigma)
        fitness = calculateFitness(neighbor)
        if fitness < best_fitness and neighbor not in tabu_list:
            best_fitness = fitness
            best_puzzle = copy.deepcopy(neighbor)
        tabu_list.append(neighbor)
    #         print('mutation2',calculateFitness(best_puzzle))
    # print('output mutation2',calculateFitness(best_puzzle))
    return best_puzzle
def localSearch3(puzzle):#模拟退火法
    temperature=1000.0
    final_temperature=0.1
    cooling_rate=0.95
    current_solution=copy.deepcopy(puzzle)
    current_fitness=calculateFitness(current_solution)
    while temperature>final_temperature:
        new_solution=copy.deepcopy(current_solution)
        operation=random.choice(['mutation1','mutation2'])
        if operation=='mutation1':
            new_solution=mutation1(new_solution,mutation_rate=1.0,sigma=1)
        else:
            new_solution=mutation2(new_solution,mutation_rate=1.0,sigma=1)
        new_fitness=calculateFitness(new_solution)
        if new_fitness<current_fitness:
            current_solution=copy.deepcopy(new_solution)
            current_fitness=new_fitness
        # else:
        #     probability=math.exp((current_fitness-new_fitness)/temperature)
        #     if random.random()<probability:
        #         current_solution=copy.deepcopy(new_solution)
        #         current_fitness=new_fitness
        temperature*=cooling_rate
    return current_solution



def build_distance_matrix(population):
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
    for idv_tile in range(Colsize*Rowsize):
        #如3142 与1423 的dis为1，3142 2872如果不匹配权重最大8
        angle=abs((puzzle1[idv_tile][2]-puzzle2[idv_tile][2])%3+(puzzle1[idv_tile][2]-puzzle2[idv_tile][2])//3)
        id_diff=puzzle1[idv_tile][0]-puzzle2[idv_tile][0]
        if id_diff !=0:
            distance=8     #set the weight
        else:
            distance=angle
    return distance
def extra_population(population_X):
    new_population=[]
    sigma=Rowsize*Colsize*0.5
    for _ in range(len(population_X)//2):
        random_parent=random.sample(range(0, len(population_X)), 5)
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
        # child1,child2=Crossover(parent1,parent2,2,2)
        # child1,child2=EdgeRecombination(parent1,parent2)
        # child1,child2=EdgeRecombination2D(parent1,parent2)
        child1=localSearch(child1,1,sigma)
        child2=localSearch(child2,1,sigma)
        child1=localSearch2(child1,1,sigma)
        child2=localSearch2(child2,1,sigma)
        # child1=mutation1(child1,1,sigma)
        # child2=mutation1(child2,1,sigma)
        # child1=mutation2(child1,1,sigma)
        # child2=mutation2(child2,1,sigma)
        new_population.append(child1)
        new_population.append(child2)
    new_population+=population_X
    return new_population[:len(population_X)]

def main():
    global OutOrIN
    OutOrIN=1
    dict_fitness_value={"4":1,"1":4}# 权值变化
    population=initialization()
    fitness_board=list(map(calculateFitness,population))
    best_fitness=min(fitness_board)
    previous_best_fitness=best_fitness
    mutation_rate=initial_mutation_rate
    sigma=initial_sigma
    Generation=0
    fitness_list=[]
    do_local_search3=0
    real_best_fitness=999
    real_best_mismatch=9999
    while fitness_board[fitness_board.index(min(fitness_board))]>0 and Generation<maxGeneration:
        distance_matrix=build_distance_matrix(population)
        # random_parent=random.sample(range(0, population_size), int(population_size*children_Percent))
        # windows=[[i,population[i]] for i in random_parent]
        # windows.sort(key=lambda x:calculateFitness(x[1]))
        
        '''
         Diversity
        '''
        # for index,parent_select in enumerate(population):
        #     new_population=[]
        #     parent1=copy.deepcopy(parent_select)
        #     distance_matrix[index].sort(key=lambda x:x[0])
        #     select_index=random.randint(0,population_size-2)
        #     parent2=copy.deepcopy(population[distance_matrix[index][select_index][1]])
        #     random_select=random.randint(0,100)
        #     if random_select<10:
        #         child1,child2=Crossover(parent1,parent2,3,3)
        #     elif random_select<40:
        #         child1,child2=EdgeRecombination(parent1,parent2)
        #     elif random_select<90:
        #         child1,child2=EdgeRecombination2D(parent1,parent2)
        #     else:
        #         child1,child2=parent1,parent2
        #     child1=mutation1(child1,mutation_rate,sigma)
        #     child2=mutation1(child2,mutation_rate,sigma)
        #     child1=mutation2(child1,mutation_rate,sigma)
        #     child2=mutation2(child2,mutation_rate,sigma)
        #     new_population.append(child1)
        #     new_population.append(child2)
        #     new_population.sort(key=lambda x:calculateFitness(x))
        #     if calculateFitness(new_population[0])<calculateFitness(parent_select):
        #         population[index]=new_population[0]
        '''
        Tournament Selection
        '''
        random_parent=random.sample(range(0, population_size), int(population_size*children_Percent))
        windows=[[i,copy.deepcopy(population[i])] for i in random_parent]
        windows.sort(key=lambda x:calculateFitness(x[1]))
        new_population=[]
        for parent_select in range(len(windows)//2):
            parent1=windows[parent_select*2][1]
            parent2=windows[parent_select*2+1][1]
            # child1,child2=Crossover(parent1,parent2,2,2)
            # child1,child2=EdgeRecombination(parent1,parent2)
            # child1,child2=EdgeRecombination2D(parent1,parent2)
            random_select=random.randint(0,100)
            if random_select<10:
                child1,child2=Crossover(parent1,parent2,3,3)
            elif random_select<40:
                child1,child2=EdgeRecombination(parent1,parent2)
            elif random_select<90:
                child1,child2=EdgeRecombination2D(parent1,parent2)
            else:
                child1,child2=parent1,parent2
            child1=localSearch(child1,mutation_rate,sigma)
            child2=localSearch(child2,mutation_rate,sigma)
            child1=localSearch2(child1,mutation_rate,sigma)
            child2=localSearch2(child2,mutation_rate,sigma)
            # child1=mutation1(child1,mutation_rate,sigma)
            # child2=mutation1(child2,mutation_rate,sigma)
            # child1=mutation2(child1,mutation_rate,sigma)
            # child2=mutation2(child2,mutation_rate,sigma)
            new_population.append(child1)
            new_population.append(child2)
        new_population+=[copy.deepcopy(population[i]) for i in random_parent]
        new_population.sort(key=lambda x:calculateFitness(x))
        for index,old_population in enumerate(windows):
            population[random_parent[index]]=copy.deepcopy(new_population[index])
        population.sort(key=lambda x:calculateFitness(x))
        # fitness_board=list(map(calculateFitness,copy.deepcopy(population)))
        # best_fitness=min(fitness_board)
        # for offspring in new_population:
        #     for index,old_population in enumerate(windows):
        #         if calculateFitness(offspring)<calculateFitness(old_population[1]):
        #             windows[index][1]=offspring
        #             break
        #-----------------------------------------------------------------------------------------------------------------
        # fitness_board=list(map(calculateFitness,population))
        # best_fitness=min(fitness_board)
        # best_individual=population[fitness_board.index(best_fitness)]
        # mismatch_Board=list(map(calculateMissmatch,copy.deepcopy(population)))
        # best_individual=localSearch(best_individual)
        # best_individual=localSearch2(best_individual)
        # best_individual=localSearch3(best_individual)
        # population[fitness_board.index(best_fitness)]=copy.deepcopy(best_individual)
        random_local_search=random.sample(range(0, population_size), population_size//2)
        for R_index in random_local_search:
            R_individual=population[R_index]
            R_individual=localSearch(R_individual)
            R_individual=localSearch2(R_individual)
            # R_individual=localSearch3(R_individual)
            population[R_index]=copy.deepcopy(R_individual)
        # print(f'bestindividual:{calculateFitness(best_individual)}')
        # fitness_board=list(map(calculateFitness,population))
        # best_fitness=min(fitness_board)
        # print(f'Best Fitness = {best_fitness}')
        fitness_board=list(map(calculateFitness,copy.deepcopy(population)))
        best_fitness=min(fitness_board)
        fitness_list.append(best_fitness)
        mismatch_Board=list(map(calculateMissmatch,population))
        if previous_best_fitness != 0:    #self-adaptive mutayion
            improvement_rate = (previous_best_fitness - best_fitness) / previous_best_fitness
        else:
            improvement_rate = 0
        mutation_rate = self_adaptive_Pm(mutation_rate,final_mutation_rate,initial_mutation_rate,improvement_rate)#self-adaptive mutayion
        sigma = self_adaptive_Pm(sigma,final_sigma,initial_sigma,improvement_rate)
        previous_best_fitness=best_fitness
        Generation+=1
        print(f'Generation {Generation}: Best Fitness = {best_fitness} :Best Mismatch = {mismatch_Board[fitness_board.index(best_fitness)]}')
        # print(mismatch_Board[:10])
        if Generation % 5 ==0:
            # print('mutation_rate',mutation_rate)
            # print('sigma',sigma)
            if best_fitness==fitness_list[-3]:
                OutOrIN=dict_fitness_value[str(OutOrIN)]
                num_replace = int(0.97 * population_size)
            else:
                num_replace = int(0.5 * population_size)
            population.sort(key=lambda x:calculateFitness(x))
            
            new_individuals = extra_population(copy.deepcopy(population[-num_replace:]))
            population[-num_replace:] = copy.deepcopy(new_individuals)
            # new_kjllation[-num_replace:] = copy.deepcopy(new_individuals)
        best_solution=population[fitness_board.index(best_fitness)]
        best_mismatch=calculateMissmatch(best_solution)
        if best_fitness<real_best_fitness or best_mismatch<real_best_mismatch:
            best_solution=population[fitness_board.index(best_fitness)]
            best_mismatch=calculateMissmatch(best_solution)
            real_best_fitness=best_fitness
            real_best_mismatch=best_mismatch
            write_file(best_solution,best_mismatch)
    format_solution(best_solution)
    best_mismatch=calculateMissmatch(best_solution)
    write_file(best_solution,best_mismatch)
if __name__ == '__main__':
    main()