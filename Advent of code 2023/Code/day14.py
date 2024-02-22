def read_input(num,is_example=False):
    lines = []
    path = "C:/Users/jorge/Desktop/UNIVERSIDAD/!Proyectos/Python/Advent of code 2023/Inputs/input" + num + ".txt"
    if is_example:
        path = "C:/Users/jorge/Desktop/UNIVERSIDAD/!Proyectos/Python/Advent of code 2023/Inputs/example" + num + ".txt"
    with open(path, 'r') as file:
        for line in file:
            lines.append(line.removeprefix("\n").removesuffix("\n"))
    return lines    

def new_grid(num_rows,num_cols,symbol):
    grid = []
    for i in range(num_rows):
        row = []
        for j in range(num_cols):
            row.append(symbol)
        grid.append(row)
    return grid


def create_grid_from_lines(lines,fill_emptyness=False,empty_symbol=" "):
    grid = []
    if fill_emptyness:
        max_ = 0
        for line in lines:
            max_ = max(max_,len(line))
        
        for i in range(len(lines)):
            lines[i] += (max_ - len(lines[i]))*empty_symbol
    
    for line in lines:
        grid.append(list(line))
    return grid

def show_grid(grid,separator=""):
    for i, line in enumerate(grid): 
        print("{:<3}".format(i) + ": ", end = "")
        for col in line:
            print(col,end=separator)
        print()

def get_grids(lines):
    grids = []
    current_grid = []
    for line in lines:
        if line != '':
            current_grid.append(line)
        else:
            grids.append(create_grid_from_lines(current_grid))
            current_grid = []
    if current_grid != '':
        grids.append(create_grid_from_lines(current_grid))
    return grids

def copy_grid(grid):
    new_grid = []
    for row in grid:
        new_grid.append(row[:])
    return new_grid

def matches(grid1,grid2):
    if len(grid1) != len(grid2):
        return False
    for i,row in enumerate(grid1):
        if len(grid1[i]) != len(grid2[i]):
            return False
        for j,elem in enumerate(grid1[i]):
            if grid1[i][j] != grid2[i][j]:
                return False
    return True

def tilt_to_dir(grid,dir):
    empty_indices = []

    #NORTH
    if dir == 0:
        for i,row in enumerate(grid):
            for j,elem in enumerate(grid[i]):
                if j >= len(empty_indices):
                    empty_indices.append(-1)
                if elem == '.':
                    if empty_indices[j] == -1:
                        empty_indices[j] = i
                elif elem == '#':
                    empty_indices[j] = -1
                else:
                    if empty_indices[j] != -1:
                        grid[i][j] = '.'
                        grid[empty_indices[j]][j] = 'O'
                        prev_index = empty_indices[j]
                        empty_indices[j] = -1
                        for k in range(prev_index,i+1):
                            if grid[k][j] == '.':
                                if empty_indices[j] == -1:
                                    empty_indices[j] = k
                            else:
                                empty_indices[j] = -1
    #SOUTH
    elif dir == 2:
         for i in range(len(grid)-1,-1,-1):
            for j,elem in enumerate(grid[i]):
                if j >= len(empty_indices):
                    empty_indices.append(-1)
                if elem == '.':
                    if empty_indices[j] == -1:
                        empty_indices[j] = i
                elif elem == '#':
                    empty_indices[j] = -1
                else:
                    if empty_indices[j] != -1:
                        grid[i][j] = '.'
                        grid[empty_indices[j]][j] = 'O'
                        #Calculate new hole
                        prev_index = empty_indices[j]
                        empty_indices[j] = -1
                        for k in range(prev_index,i-1,-1):
                            if grid[ k][j] == '.':
                                if empty_indices[j] == -1:
                                    empty_indices[j] = k
                            else:
                                empty_indices[j] = -1
    #WEST
    elif dir == 1:
        for i,row in enumerate(grid):
            for j,elem in enumerate(grid[i]):
                if i >= len(empty_indices):
                    empty_indices.append(-1)
                if elem == '.':
                    if empty_indices[i] == -1:
                        empty_indices[i] = j
                elif elem == '#':
                    empty_indices[i] = -1
                else:
                    if empty_indices[i] != -1:
                        grid[i][j] = '.'
                        grid[i][empty_indices[i]] = 'O'
                        prev_index = empty_indices[i]
                        empty_indices[i] = -1
                        for k in range(prev_index,j+1):
                            if grid[i][k] == '.':
                                if empty_indices[i] == -1:
                                    empty_indices[i] = k
                            else:
                                empty_indices[i] = -1
    
    #EAST
    elif dir == 3:
        for i,row in enumerate(grid):
            for j in range(len(grid[i])-1,-1,-1):
                elem = grid[i][j]
                if i >= len(empty_indices):
                    empty_indices.append(-1)
                if elem == '.':
                    if empty_indices[i] == -1:
                        empty_indices[i] = j
                elif elem == '#':
                    empty_indices[i] = -1
                else:
                    if empty_indices[i] != -1:
                        grid[i][j] = '.'
                        grid[i][empty_indices[i]] = 'O'
                        prev_index = empty_indices[i]
                        empty_indices[i] = -1
                        for k in range(prev_index,j-1,-1):
                            if grid[i][k] == '.':
                                if empty_indices[i] == -1:
                                    empty_indices[i] = k
                            else:
                                empty_indices[i] = -1

def get_rock_load(grid):
    total_load = 0
    for i,row in enumerate(grid):
        for j,elem in enumerate(grid[i]):
            if elem == 'O':
                total_load += len(grid)-i
    return total_load

def contains(memo,grid):
    for index,g in memo:
        if matches(g,grid):
            return True, index
    return False, -1

def main():
    lines = read_input("14",True)
    grid = create_grid_from_lines(lines)
    # show_grid(grid)
    cycle = 0
    num_cycles = 1000000000
    memo = []
    while cycle < num_cycles:
        for k in range(4):
            tilt_to_dir(grid,k)
        print("Cycle:",cycle)
        show_grid(grid)
        contained,index = contains(memo,grid)
        if not contained:
            memo.append((cycle,copy_grid(grid)))
        else:
            
            cycle_length = cycle-index
            cycle = cycle + ((num_cycles-cycle)//cycle_length) * cycle_length
            print(cycle,index,cycle_length,num_cycles)
            
        cycle +=1
        if (cycle+1) % 1000 == 0:
            print(cycle+1,"/",num_cycles)
    
    print(get_rock_load(grid))

#Calling
main()
