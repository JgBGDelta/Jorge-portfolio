def read_input(num,is_example=False):
    lines = []
    path = "C:/Users/jorge/Desktop/UNIVERSIDAD/!Proyectos/Python/Advent of code 2023/Inputs/input" + num + ".txt"
    if is_example:
        path = "C:/Users/jorge/Desktop/UNIVERSIDAD/!Proyectos/Python/Advent of code 2023/Inputs/example" + num + ".txt"
    with open(path, 'r') as file:
        for line in file:
            lines.append(line.removeprefix("\n").removesuffix("\n"))
    return lines    

def create_grid_from_lines(lines,complete_emptyness,empty_symbol=" "):
    grid = []
    # if complete_emptyness:
    #     max_ = 0
    #     for line in lines:
    #         max_ = max(max_,len(line))
        
    #     for i in range(len(lines)):
    #         lines[i] += (max_ - len(lines[i]))*empty_symbol
    for line in lines:
        grid.append(list(line.replace(" ",empty_symbol)))
    return grid

def show_grid(grid,separator=""):
    for i, line in enumerate(grid): 
        print("{:<3}".format(i) + ": ", end = "")
        for col in line:
            print(col,end=separator)
        print()



def is_adjacent_to_symbol(coord, grid):
    i,j = coord
    directions = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]

    for dir in directions:

        #New directions
        new_i = i +dir[0]
        new_j = j +dir[1]
        
        #Check if in bounds
        if new_i >= 0 and new_i < len(grid) and new_j>=0 and new_j<len(grid[i]):
            char = grid[new_i][new_j]

            #If is symbol
            if char != "." and not char.isdigit():
                return True, (new_i,new_j)
            
    return False, (-1,-1)


def main():
    lines = read_input("3",is_example=False)
    grid = create_grid_from_lines(lines,False)
    show_grid(grid)

    adjacent_numbers = []
    for i,row in enumerate(grid):
        is_adjacent = False
        current_num_str = ""
        next_engines = []
        adjacent_numbers_row = []
        for j,char in enumerate(row):

            if char.isdigit():
                current_num_str += char
                adjacent, (new_i,new_j) = is_adjacent_to_symbol((i,j),grid)

                if adjacent and grid[new_i][new_j] == "*":
                    if not (new_i,new_j) in next_engines:
                        next_engines.append((new_i,new_j))

                if adjacent and not is_adjacent:
                    is_adjacent = adjacent

            else:
                if current_num_str != "" and is_adjacent:
                    adjacent_numbers_row.append(int(current_num_str))
                    for pos in next_engines:
                        if pos in ENGINES:
                            ENGINES[pos].append(int(current_num_str))
                        else:
                            ENGINES[pos] = [int(current_num_str)] 
                    next_engines = []
                    is_adjacent = False
                current_num_str = ""

        if current_num_str != "" and is_adjacent:
            adjacent_numbers_row.append(int(current_num_str))
            for pos in next_engines:
                if pos in ENGINES:
                    ENGINES[pos].append(int(current_num_str))
                else:
                    ENGINES[pos] = [int(current_num_str)] 
            next_engines = []

        adjacent_numbers.append(adjacent_numbers_row)

    # show_grid(adjacent_numbers,separator=" ")

    print(ENGINES)
    power_sum = 0
    for key in ENGINES:
        if len(ENGINES[key]) == 2:
            power_sum += ENGINES[key][0] * ENGINES[key][1]
    print(power_sum)

    total_sum = 0
    for row in adjacent_numbers:
        total_sum += sum(row)
    print(total_sum)
                    


ENGINES = {} # Coord - List of adjacent nums
main()
