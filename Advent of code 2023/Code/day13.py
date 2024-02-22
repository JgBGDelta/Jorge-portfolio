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

def matchRows(row1,row2):
    diffs = 0
    pos = -1
    for i,char in enumerate(row1):
        if row1[i] != row2[i]:
            diffs +=1
            pos = i
    if diffs == 0:
        return True,diffs,pos
    else:
        return False,diffs,pos


def check_horizontal_symmetry(grid):
    symmetry_axis = -1
    axis_diffs = [(0,-1)]*len(grid)

    for axis,row in enumerate(grid):

        symmetric = True
        index = 1
        reflection_count = axis + 1

        while reflection_count > 0 and axis+index < len(grid):
            row1,row2 = grid[axis-index+1],grid[axis+index]

            match,diffs,pos = matchRows(row1,row2)
            if axis_diffs[axis][0] == 0 and diffs == 1:
                axis_diffs[axis] = (diffs,(axis,axis-index+1,axis+index,pos))
            elif diffs != 0:
                axis_diffs[axis] = (axis_diffs[axis][0] + diffs,-1)
            if not match:
                symmetric = False
            index += 1
            reflection_count -= 1

        if symmetric and axis+1 < len(grid):
            symmetry_axis = axis

    
    return symmetry_axis,axis_diffs

def rotate_grid(grid):
    new_grid = []
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if j >= len(new_grid):
                new_grid.append([])
            new_grid[j].append(grid[i][j])
    return new_grid

def main():
    lines = read_input("13",True)
    grids = get_grids(lines)
    
    total = 0
    for grid in grids[:2]:
        total += simulate_symmetry(grid)[0]
    print(total)

def simulate_symmetry(grid):
    total = 0
    symmetry_axis = [-1,-1]
    # Horizontal
    row_axis,axis_diffs_rows = check_horizontal_symmetry(grid)
    show_grid(grid,separator=" ")
    print()
    if row_axis != -1:
        symmetry_axis[0] = row_axis
        total += 100 * (row_axis+1)

    # Vertical
    v_grid = rotate_grid(grid)
    col_axis,axis_diffs_cols = check_horizontal_symmetry(v_grid)
    show_grid(v_grid,separator=" ")
    print()
    if col_axis != -1:
        symmetry_axis[1] = col_axis
        total += col_axis + 1

    return total,symmetry_axis,axis_diffs_rows,axis_diffs_cols

def main2():
    lines = read_input("13",False)
    grids = get_grids(lines)
    
    total = 0
    changed_total = 0
    for i,grid in enumerate(grids):
        total_from_sim,symmetry_axis,axis_diffs_rows,axis_diffs_cols = simulate_symmetry(grid)
        total += total_from_sim

        if symmetry_axis == [-1,-1]:
            print("Error: No se encontró simetría")
        #Change
        print(symmetry_axis)
        print(axis_diffs_rows)
        print(axis_diffs_cols)
        change_pos = None
        new_axis = [-1,-1]
        
        for diff,info in axis_diffs_rows:
            if diff == 1:
                if change_pos != None:
                    raise "Error: Detectados dos smudges."
                change_pos = [[info[1],info[3]],[info[1],info[2]]]
                new_axis[0] = info[0]
        if change_pos == None:
            for diff,info in axis_diffs_cols:
                if diff == 1:
                    if change_pos != None:
                        raise "Error: Detectados dos smudges."
                    change_pos = [[info[3],info[1]],[info[2],info[1]]]
                    new_axis[1] = info[0]
        
        print(change_pos,new_axis)

        if change_pos == None and symmetry_axis != [-1,-1]:
            raise "Error: No hay change_pos, es decir, no se ha encontrado el smudge"

        #Si el axis es distinto
        if (symmetry_axis[0],symmetry_axis[1]) != (new_axis[0],new_axis[1]):
            if new_axis[0] != -1:
                changed_total += 100 * (new_axis[0]+1)
            if new_axis[1] != -1:
                changed_total += (new_axis[1]+1)
        elif symmetry_axis != [-1,-1]:
            raise "Error: La nueva simetría es igual a la antigua"
            pass

                
            

        
    print(total)
    print(changed_total)
  

#Calling
main2()
