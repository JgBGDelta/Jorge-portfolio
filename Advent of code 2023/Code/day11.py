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


def create_grid_from_lines(lines,complete_emptyness=False,empty_symbol=" "):
    grid = []
    if complete_emptyness:
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

def get_empty_rows_and_cols(grid):
    empty_rows = []
    empty_cols = []

    is_col_empty = [True]*len(grid[0])

    for i, row in enumerate(grid):
        empty_row = True
        for j,elem in enumerate(grid[i]):
            if elem == "#":
                empty_row = False
                if is_col_empty[j]:
                    is_col_empty[j] = False
        if empty_row:
            if len(empty_rows) == 0:
                empty_rows.append([i,1])
            elif sum(empty_rows[-1]) == i:
                empty_rows[-1][1] += 1
            else:
                empty_rows.append([i,1])

    for i,col in enumerate(is_col_empty):
        if col:
            if len(empty_cols) == 0:
                empty_cols.append([i,1])
            elif sum(empty_cols[-1]) == i:
                empty_cols[-1][1] += 1
            else:
                empty_cols.append([i,1])

    return empty_rows,empty_cols

def get_expanded_grid(grid):
    empty_rows,empty_cols = get_empty_rows_and_cols(grid)
    
    print(empty_rows)
    print(empty_cols)
    
    multiplier = 3000

    row_counter = 0
    for row,num in empty_rows:
        for k in range(num*(multiplier-1)):
            grid.insert(row+row_counter+1,list(['.']*len(grid[0])))
            row_counter+=1
    
    col_counter = 0
    for col,num in empty_cols:
        for i,row in enumerate(grid):
            for k in range(num*(multiplier-1)):
                grid[i].insert(col+col_counter,'.')
        col_counter+=num*(multiplier-1)

    return grid

def get_shortest_distances(galaxies):
    shortest_distances = []
    shortest_distances_pairs = []
    for index in range(len(galaxies)):
        for pair in range(index+1,len(galaxies)):
            dist = abs(galaxies[index][0] - galaxies[pair][0]) + abs(galaxies[index][1] - galaxies[pair][1])
            shortest_distances_pairs.append([(index+1,pair+1),dist])
            shortest_distances.append(dist)
    return shortest_distances

def sort_by_subelement(galaxies,subelement_index):
    return sorted(galaxies, key=lambda x: x[subelement_index])

def main():
    lines = read_input("11",True)
    grid = create_grid_from_lines(lines)
    show_grid(grid)
    grid = get_expanded_grid(grid)
    show_grid(grid)
    galaxies = []
    for i,row in enumerate(grid):
        for j,elem in enumerate(grid[i]):
            if elem == "#":
                galaxies.append((i,j))
    shortest_distances = get_shortest_distances(galaxies)
    print(sum(shortest_distances))


def main2():
    lines = read_input("11",False)
    grid = create_grid_from_lines(lines)
    galaxies = []
    for i,row in enumerate(grid):
        for j,elem in enumerate(grid[i]):
            if elem == "#":
                galaxies.append([i,j])
    empty_rows,empty_cols = get_empty_rows_and_cols(grid)
    print(empty_rows,"/",empty_cols)
    print(galaxies)

    space_multiplier = 1000000

    row_counter = 0
    for row,num in empty_rows:
        for galaxy in galaxies:
            if galaxy[0] >= row+row_counter:
                galaxy[0] += num*(space_multiplier-1)
        row_counter+=num*(space_multiplier-1)
    
    galaxies = sort_by_subelement(galaxies,1)

    col_counter = 0
    for col,num in empty_cols:
        for galaxy in galaxies:
            if galaxy[1] >= col+col_counter:
                galaxy[1] += num*(space_multiplier-1)
        col_counter+=num*(space_multiplier-1)

    galaxies = sort_by_subelement(galaxies,0)
    print(galaxies)
    
    shortest_distances = get_shortest_distances(galaxies)
    print(sum(shortest_distances))





#Calling
main2()
