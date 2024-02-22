def read_input(num,is_example=False):
    lines = []
    path = "C:/Users/jorge/Desktop/UNIVERSIDAD/!Proyectos/Python/Advent of code 2023/Inputs/input" + num + ".txt"
    if is_example:
        path = "C:/Users/jorge/Desktop/UNIVERSIDAD/!Proyectos/Python/Advent of code 2023/Inputs/example" + num + ".txt"
    with open(path, 'r') as file:
        for line in file:
            lines.append(line.removeprefix("\n").removesuffix("\n"))
    return lines    

def create_grid_from_lines(lines):
    grid = []
    max_ = 0
    symbol_for_empty = "|"
    for line in lines:
        max_ = max(max_,len(line))
    
    for i in range(len(lines)):
        lines[i] += (max_ - len(lines[i]))*symbol_for_empty
    
    for line in lines:
        grid.append(list(line.replace(" ",symbol_for_empty)))
    return grid

def show_grid(grid):
    for i, line in enumerate(grid): 
        print("{:<3}".format(i) + ": ", end = "")
        for j,col in enumerate(line):
            print(col,end="")
        print()
    

def getInitPosition(grid):
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == ".":
                return [i,j]

def getInstructions(line):
    instructions = []
    current_num = ""
    for char in line:
        if not str(char).isdigit():
            instructions.append(current_num)
            current_num = ""
            instructions.append(char)
        else:
            current_num += char
    return instructions

def get_next_pos(grid,current_pos,dir):
    new_pos = None
    next_pos = current_pos
    recursion_limit = 101
    while(new_pos == None and not recursion_limit <= 0):
        next_pos = getSimplePosAtDir(next_pos,dir)
        
        #Loop around   
        print(current_pos,next_pos,end="")
        if next_pos[0] >= len(grid):
            next_pos[0] = 0
        elif next_pos[0] < 0:
            next_pos[0] = len(grid)-1
        elif next_pos[1] >= len(grid[next_pos[0]]):
            next_pos[1] = 0
        elif next_pos[1] < 0:
            next_pos[1] = len(grid[next_pos[0]]) -1
        print(next_pos,end="")

        if 0 <= next_pos[0] < len(grid) and 0 <= next_pos[1] < len(grid[next_pos[0]]):
            if grid[next_pos[0]][next_pos[1]] == "." or grid[next_pos[0]][next_pos[1]] == "#":
                new_pos = next_pos
                print("done",end="")
            print(grid[next_pos[0]][next_pos[1]],end="")
            print("yes")
        else:
            print("nope")
        
        recursion_limit -=1

    print("Has new_pos")
    if(recursion_limit <= 0):
        print("Recursion limit reached at get_next_pos")
    return new_pos





def rotate(text_dir,current_dir):
    if text_dir == "R":
        current_dir += 1
    elif text_dir == "L":
        current_dir -=1
    current_dir = current_dir%4
    return current_dir

def getSimplePosAtDir(current_pos,dir):
    return [current_pos[0] + directions[dir][0], current_pos[1] + directions[dir][1]]
    

def function():
    lines = read_input("22",True)
    instructions = getInstructions(lines[-1])
    grid = create_grid_from_lines(lines[:-2])
    path_grid = create_grid_from_lines(lines[:-2])
    arrows = [">","v","<","^"]
    counter = 0

    pos = getInitPosition(grid)
    dir = 0 # 0: right, 1: down, 2: left, 3: up

    path_grid[pos[0]][pos[1]] = arrows[dir]
    

    instruction_count_limit = 1000
    #Read instructions
    for instr in instructions:
        if instr.isdigit():
            for i in range(int(instr)):
                next_pos = get_next_pos(grid,pos,dir)
                next_symbol = grid[next_pos[0]][next_pos[1]]
                if next_symbol == ".":
                    pos = next_pos
                else: #Es un muro #
                    break
                path_grid[pos[0]][pos[1]] = arrows[dir]
                counter += 1
                # path_grid[pos[0]][pos[1]] = counter%10
                    
        else:
            dir = rotate(instr,dir)
            path_grid[pos[0]][pos[1]] = arrows[dir]

        instruction_count_limit -= 1
        if instruction_count_limit == 0:
            break
    
    show_grid(path_grid)
    print(instr,pos,dir)
    row = pos[0] + 1
    col = pos[1] + 1
    print(1000*row + 4*col + dir)


#Calling
directions = [(0,1),(1,0),(0,-1),(-1,0)]
function()
