def read_input(num,is_example=False):
    lines = []
    path = "C:/Users/jorge/Desktop/UNIVERSIDAD/!Proyectos/Python/Advent of code 2023/Inputs/input" + num + ".txt"
    if is_example:
        path = "C:/Users/jorge/Desktop/UNIVERSIDAD/!Proyectos/Python/Advent of code 2023/Inputs/example" + num + ".txt"
    with open(path, 'r') as file:
        for line in file:
            lines.append(line.removeprefix("\n").removesuffix("\n"))
    return lines    

def create_grid_from_lines(lines,complete_emptyness=False,empty_symbol=" "):
    grid = []
    if complete_emptyness:
        max_ = 0
        for line in lines:
            max_ = max(max_,len(line))
        
        for i in range(len(lines)):
            lines[i] += (max_ - len(lines[i]))*empty_symbol
    
    for line in lines:
        grid.append(line)
    return grid

def show_grid(grid,separator=" ",actual_pos = None):
    for i, line in enumerate(grid): 
        print("{:<1}".format(i) + ": ", end = "")
        for j,col in enumerate(line):
            if (i,j) != actual_pos:
                print(col,end=separator)
            else:
                print('x',end=separator)
        print()




def main():
    lines = read_input("10",is_example=False)
    grid = create_grid_from_lines(lines)
    start_pos = None
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] == "S":
                start_pos = (i,j)
    show_grid(grid)
    # corner, incoming dir, result dir
    corners = {"L":[(1,0),(0,1)],"F":[(-1,0),(0,1)],"J":[(1,0),(0,-1)],"7":[(-1,0),(0,-1)]}
    dirs = [(1,0),(-1,0),(0,1),(0,-1)]
    main_loop_found = False
    loop_length = 0
    while not main_loop_found:
        for dir in dirs:
            connected = True
            pos = start_pos
            direction = dir
            current_dist = 0
            while connected:
                pos = (pos[0] + direction[0],pos[1] + direction[1])

                if pos[0] < 0 or pos[1] < 0 or pos[0] >= len(grid) or pos[1] >= len(grid[0]):
                    connected = False
                    break
                if pos == start_pos:
                    main_loop_found = True
                    loop_length = current_dist
                    break

                grid_elem = grid[pos[0]][pos[1]]
                if grid_elem == '|':
                    #Vertical incoming dir
                    if direction == (1,0) or direction == (-1,0):
                        pass
                    else:
                        connected = False
                elif grid_elem == '-':
                    #Horizontal incoming dir
                    if direction == (0,1) or  direction == (0,-1):
                        pass
                    else:
                        connected = False
                elif grid_elem in corners:
                    #Match coming dir
                    if corners[grid_elem][0] == direction:
                        direction = corners[grid_elem][1]
                    elif corners[grid_elem][1] == (-direction[0],-direction[1]):
                        direction = (-corners[grid_elem][0][0],-corners[grid_elem][0][1])
                    else:
                        connected = False
                else:
                    connected = False
                    pass
                # show_grid(grid,actual_pos=pos)
                current_dist += 1

            #for
            if main_loop_found:
                break
    
    print(loop_length//2 + 1)

#Calling
main()
