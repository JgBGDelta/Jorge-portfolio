import multiprocessing as mp
import time

def read_input(num,is_example=False):
    lines = []
    path = "C:/Users/jorge/Desktop/UNIVERSIDAD/!Proyectos/Python/Advent of code 2023/Inputs/input" + num + ".txt"
    if is_example:
        path = "C:/Users/jorge/Desktop/UNIVERSIDAD/!Proyectos/Python/Advent of code 2023/Inputs/example" + num + ".txt"
    with open(path, 'r') as file:
        for line in file:
            lines.append(line.removeprefix("\n").removesuffix("\n"))
    return lines    


# Global variables
# manager = mp.Manager()

energyzed = None
grid = None
rep_grid = None
beams = None
visited = None
beam_id_count = 0
DIRS = {'R':(0,1),'D':(1,0),'L':(0,-1),'U':(-1,0)}
REDIRECTIONS = { '\\': ['RD','LU'],'/':['RU','LD'],'|':['RUD','LUD'],'-':['URL','DRL']}

multiprocesses = []

#region Grid functions
def new_grid(num_rows,num_cols,symbol,emptyLists=False):
    grid = []
    for i in range(num_rows):
        row = []
        for j in range(num_cols):
            if not emptyLists:
                row.append(symbol)
            else:
                row.append([])
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

#endregion

def get_num_energyzed(energyzed_grid):
    num = 0
    for i,row in enumerate(energyzed_grid):
        for j,el in enumerate(energyzed_grid[i]):
            if el == '#':
                num += 1
    return num

def simulate_beam(id,iter):
    global energyzed,grid,rep_grid,beams,beam_id_count
    pos,dir = beams[id]
    #Energyze current place
    energyzed[pos[0]][pos[1]] = '#'
    #Simulate beam
    next_elem = grid[pos[0]][pos[1]]
    #Simulate direction change or beam duplication
    new_beams_ids = []
    destroyed_beam_ids = []
    if next_elem == '.':
        pass
    elif next_elem == "\\" or next_elem  == '/':
        redirs = REDIRECTIONS[next_elem]
        for redir in redirs:
            if dir == redir[0]:
                beams[id][1],dir = redir[1],redir[1]
                break
            if dir == redir[1]:
                beams[id][1],dir = redir[0],redir[0]
                break
    elif next_elem == '|' or next_elem == '-':
        redirs = REDIRECTIONS[next_elem]
        for redir in redirs:
            if dir == redir[0]:
                beams[id][1],dir = redir[1],redir[1]
                beam_id_count += 1
                beams[beam_id_count] = [[pos[0],pos[1]],redir[2]]
                new_beams_ids.append(beam_id_count)
                break
    else:
        raise Exception(f"Error: El caracter encontrado no coincide: {next_elem}")

    #Step ligtt for the current beam
    next_pos = [pos[0] + DIRS[dir][0],pos[1] + DIRS[dir][1]]
    if 0 <= next_pos[0] < len(grid) and 0<= next_pos[1] < len(grid[0]):
        beams[id][0] = next_pos
        rep_grid[next_pos[0]][next_pos[1]] = iter%10
        if dir in visited[next_pos[0]][next_pos[1]]:
            beams.pop(id)
        else:
            visited[next_pos[0]][next_pos[1]].append(dir)
    else:
        beams.pop(id)
    #Step all newly created ligths
    for new_id in new_beams_ids:
        new_pos,new_dir = beams[new_id]
        next_pos = [new_pos[0] + DIRS[new_dir][0],new_pos[1] + DIRS[new_dir][1]]
        if 0 <= next_pos[0] < len(grid) and 0<= next_pos[1] < len(grid[0]):
            beams[new_id][0] = next_pos
            rep_grid[next_pos[0]][next_pos[1]] = iter%10
            if new_dir in visited[next_pos[0]][next_pos[1]]:
                beams.pop(new_id)
            else:
                visited[next_pos[0]][next_pos[1]].append(new_dir)
        else:
            beams.pop(new_id)

    # show_grid(rep_grid)
    # show_grid(energyzed)
    
    return new_beams_ids

def process_ligth_beam(id,iterations):
    """ Función ejecutada por cada proceso individual que simula el movimiento del rayo de luz
    indicado y modifica la energyzed grid global"""
    print(beams)
    for it in range(iterations):
        print(f"Proceso {0}, iteracion: {it}")
        if not id in beams:
            break
        new_beams = simulate_beam(id,it)
        if len(new_beams) > 0:
            for new_id in new_beams:
                print(f'Wanna create {new_id}')
                # process = mp.Process(target=process_ligth_beam,args=(new_id,(iterations-it)))
                # multiprocesses.append(process)
                # process.start()
    print(f"Proceso con id {id} finalizado")


def main(startPos,startDir,lines,grid):
    #Data creation
    global rep_grid,energyzed,beams,visited
    rep_grid = create_grid_from_lines(lines)
    beams = {0:[startPos,startDir]} # id: [(posx,posy) , dir R,D,L,U]
    beam_id_count = 1
    rep_grid[0][0] = '#'
    energyzed = new_grid(len(grid),len(grid[0]),'.')
    visited = new_grid(len(grid),len(grid[0]),0,emptyLists=True)

    #Show grids
    # show_grid(grid)
    # show_grid(energyzed)
    # show_grid(visited)

    #Process creation and start
    iterations = 1000
    print(beams)
    if False:
        first_process = mp.Process(target=process_ligth_beam,args=(0,iterations))
        multiprocesses.append(first_process)
        first_process.start()
        first_process.join()
    else:
        last_state = []
        match_counter = 0
        dobule_break = False
        for it in range(iterations):
            if dobule_break:
                break
            beam_ids = list(beams.keys())[:]
            for id in beam_ids:
                if id in beams:
                    #Simulate the beam of ligth
                    simulate_beam(id,it)
                    if matches(energyzed,last_state):
                        match_counter+=1
                    else:
                        match_counter = 0
                        last_state = copy_grid(energyzed)
                    if match_counter >= 1000:
                        print("Iteraciones finales:",it)
                        dobule_break = True
                        break
                    # show_grid(rep_grid)
            if it%1000 == 0:
                print("Iteracion:",it)
                pass
    
    # show_grid(rep_grid)
    # show_grid(energyzed)
    num_e = get_num_energyzed(energyzed)
    print(num_e)
    return num_e

#Calling
if __name__ == '__main__':
    lines = read_input("16",False)
    grid = create_grid_from_lines(lines)
    startTime = time.time()
    max_energyzed = 0
    best_start = None
    for i in range(len(grid)):
        #Left side
        num_e = main([i,0],'R',lines,grid)
        if num_e > max_energyzed:
            max_energyzed = num_e
            best_start = [i,0]
        #Right side
        num_e2 = main([i,len(grid)-1],'L',lines,grid)
        if num_e2 > max_energyzed:
            max_energyzed = num_e2
            best_start = [i,len(grid)-1]

    for j in range(len(grid[0])):
        #Top side
        num_e = main([0,j],'D',lines,grid)
        if num_e > max_energyzed:
            max_energyzed = num_e
            best_start = [i,0]
        #Bottom side
        num_e2 = main([len(grid)-1,j],'L',lines,grid)
        if num_e2 > max_energyzed:
            max_energyzed = num_e2
            best_start = [len(grid)-1,j]
    
    print("Time: ", time.time()-startTime)
    print(max_energyzed,best_start)
    # print(main([0,3],'D',lines,grid))

