def read_input(num,is_example=False):
    lines = []
    path = "C:/Users/jorge/Desktop/UNIVERSIDAD/!Proyectos/Python/Advent of code 2023/Inputs/input" + num + ".txt"
    if is_example:
        path = "C:/Users/jorge/Desktop/UNIVERSIDAD/!Proyectos/Python/Advent of code 2023/Inputs/example" + num + ".txt"
    with open(path, 'r') as file:
        for line in file:
            lines.append(line.removeprefix("\n").removesuffix("\n"))
    return lines    

def get_subsequences(subseqs):
    reached_zero = False
    i = 0
    while not reached_zero and i < len(subseqs):
        current_seq = subseqs[i]
        new_seq = []
        reached_zero = True
        for k in range(1,len(current_seq)):
            diff = current_seq[k]-current_seq[k-1]
            new_seq.append(diff)
            if diff != 0:
                reached_zero = False
        subseqs.append(new_seq)
        i += 1

def main():
    lines = read_input("9",False)
    total = 0
    for line in lines:
        seq = [int(x) for x in line.split()]
        subseqs = [seq]
        get_subsequences(subseqs)
        #1
        # subseqs[-1].append(0)
        # for depth in range(len(subseqs)-2,-1,-1):
        #     subseqs[depth].append(subseqs[depth][-1] + subseqs[depth+1][-1])
        # next_value = subseqs[0][-1]

        #2
        subseqs[-1].insert(0,0)
        for depth in range(len(subseqs)-2,-1,-1):
            subseqs[depth].insert(0, subseqs[depth][0] - subseqs[depth+1][0])
            next_value = subseqs[0][0]
        total += next_value
        print(subseqs)
    print(total)

#Calling
main()
