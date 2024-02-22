def read_input(num,is_example=False):
    lines = []
    path = "C:/Users/jorge/Desktop/UNIVERSIDAD/!Proyectos/Python/Advent of code 2023/Inputs/input" + num + ".txt"
    if is_example:
        path = "C:/Users/jorge/Desktop/UNIVERSIDAD/!Proyectos/Python/Advent of code 2023/Inputs/example" + num + ".txt"
    with open(path, 'r') as file:
        for line in file:
            lines.append(line.removeprefix("\n").removesuffix("\n"))
    return lines    

def HASH_algorythm(chain):
    current_value = 0
    for char in chain:
        current_value += ord(char)
        current_value *= 17
        current_value = current_value%256
    return current_value

def index(lens, box):
    for i,pair in enumerate(box):
        label,focal = pair
        if label == lens[0]:
            return i
    return -1

def simulate_operations(strings,boxes):
    results = {}
    for chain in strings:
        label,operator,focal = '','',''
        if '=' in chain:
            operator = '='
            label,focal = chain.split('=')
        elif '-' in chain:
            operator = '-'
            label = chain[:-1]

        #Obtener box_num
        if label in results:
            box_num = results[label]
        else:
            box_num = HASH_algorythm(label)
            results[label] = box_num
        
        #Si no existe caja se crea
        if not box_num in boxes:
            boxes[box_num] = []

        pos = index((label,focal),boxes[box_num])
        if operator == '=':
            # Si ya se encuentra en la caja, se cambia su focal length nada más
            if pos >= 0:
                boxes[box_num][pos] = (label,focal)
            # Si no está en la caja, se añade al final de la lista de lentes
            else:
                boxes[box_num].append((label,focal))
        else: #-
            #Si la lente se encuentra en la caja se saca y el resto avanzan hacia delante (pop)
            if pos >= 0:
                boxes[box_num].pop(pos)

def get_focusing_power(boxes):
    focusing_power = 0
    for box in boxes:
        for i,lens in enumerate(boxes[box]):
            temp = (box+1) * (i+1) * int(lens[1])
            focusing_power += temp
    return focusing_power


def main():
    lines = read_input("15",False)
    strings = lines[0].split(',')

    boxes = {}
    simulate_operations(strings,boxes)
    print(boxes)
    focus_power = get_focusing_power(boxes)
    print(focus_power)
    
    
#Calling
main()
