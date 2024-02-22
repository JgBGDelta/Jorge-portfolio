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
    lines = read_input("6",False)
    times = [int(x) for x in lines[0].split(":")[1].split()]
    distances = [int(x) for x in lines[1].split(":")[1].split()]
    
    result = 1
    for i,time in enumerate(times):
        num_ways_surpass_record = 0
        for k in range(time):
            dist = k*(time-k)
            if dist > distances[i]:
                num_ways_surpass_record += 1
        result *= num_ways_surpass_record
    print(result)


def main2():
    lines = read_input("6",True)
    times = [int(str(lines[0].split(":")[1]).replace(" ",""))]
    distances = [int(str(lines[1].split(":")[1]).replace(" ",""))]
    print(times)
    print(distances)
    
    result = 1
    for i,time in enumerate(times):
        num_ways_surpass_record = 0
        for k in range(time):
            dist = k*(time-k)
            if dist > distances[i]:
                num_ways_surpass_record += 1
        result *= num_ways_surpass_record
    print(result)

    print(times[0]/2)
    print(times[0]*times[0]/4)

#Calling
main2()
