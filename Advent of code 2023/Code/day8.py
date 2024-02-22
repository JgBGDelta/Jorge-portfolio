import math


def read_input(num,is_example=False):
    lines = []
    path = "C:/Users/jorge/Desktop/UNIVERSIDAD/!Proyectos/Python/Advent of code 2023/Inputs/input" + num + ".txt"
    if is_example:
        path = "C:/Users/jorge/Desktop/UNIVERSIDAD/!Proyectos/Python/Advent of code 2023/Inputs/example" + num + ".txt"
    with open(path, 'r') as file:
        for line in file:
            lines.append(line.removeprefix("\n").removesuffix("\n"))
    return lines    

def main():
    lines = read_input("8",False)
    instructions = lines[0]
    graph = {}
    for line in lines[2:]:
        elem, children = line.split(' = ')
        children = children.split(', ')
        children[0] = children[0].removeprefix('(')
        children[1] = children[1].removesuffix(')')
        graph[elem] = children
    
    zzz_found = False
    inst_counter = 0
    conversion = {'L':0,'R':1}
    current = 'AAA'
    step_counter = 0
    while not zzz_found:
        inst = instructions[inst_counter]

        current = graph[current][conversion[inst]]
        if current == 'ZZZ':
            zzz_found = True

        inst_counter += 1
        step_counter += 1
        if inst_counter >= len(instructions):
            inst_counter = 0

    print(step_counter)        

def main2():
    lines = read_input("8",False)
    instructions = lines[0]
    graph = {}
    for line in lines[2:]:
        elem, children = line.split(' = ')
        children = children.split(', ')
        children[0] = children[0].removeprefix('(')
        children[1] = children[1].removesuffix(')')
        graph[elem] = children
    
    all_z = False
    inst_counter = 0
    conversion = {'L':0,'R':1}
    currents = []
    for node in graph:
        if node[2] == "A":
            currents.append(node)

    cycles = {} # elem : inst_count, step_count
    partial_cycles = []
    for i in range(len(currents)):
        partial_cycles.append({})
    completed_cycles = []
    step_counter = 0
    cycles_found = False

    while not all_z and not cycles_found and step_counter < 1000000000:
        inst = instructions[inst_counter]
        all_z = True
        for i,current in enumerate(currents):
            currents[i] = graph[current][conversion[inst]]
            if currents[i][2] != 'Z':
                all_z = False
            else:
                if not i in completed_cycles:
                    if currents[i] in partial_cycles[i]:
                        # if partial_cycles[i][currents[i]][0] == inst_counter:
                        completed_cycles.append(i)
                        cycles[currents[i]] = partial_cycles[i][currents[i]]
                    else:
                        partial_cycles[i][currents[i]] = [inst_counter,step_counter+1]
                    
        inst_counter += 1
        step_counter +=1
        if len(completed_cycles) == len(currents):
            cycles_found = True
        if inst_counter >= len(instructions):
            inst_counter = 0
    
    print(cycles)
    nums = []
    for cycle in cycles:
        nums.append(cycles[cycle][1])
    print(math.lcm(*nums))

#Calling
main2()
