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

def get_maps(lines):
    maps = []
    new_map = []
    for line in lines[2:]:
        if line == "":
            if len(new_map) != 0:
                maps.append(new_map)
                new_map = []
        else:
            final_line = None
            if line[0].isdigit():
                final_line = [int(x) for x in line.split(" ")]
                new_map.append(final_line)
            # else:
            #     temp = line.split("-")
            #     final_line = [temp[0],temp[2].split(" ")[0]]
    if len(new_map) > 0:
        maps.append(new_map)
    return maps


def convertFromMap(value,map):
    """ int, list[list] -> Any"""
    for conversion_line in map:
        dest_start, origin_start, step = conversion_line
        #If in range of conversion
        if origin_start <= value <= origin_start + step:
            return dest_start + (value-origin_start)
    return value
    


def main():
    lines = read_input("5",False)

    seed_ranges = [int(x) for x in lines[0].split(": ")[1].split(" ")]
    maps = get_maps(lines)
    seeds = []
    
    for i in range(0,len(seed_ranges),2):
        for k in range(seed_ranges[i+1]):
            seeds.append(seed_ranges[i]+k)
        print("Num de seed_range:",i,"/",len(seed_ranges))

    locations = []
    for i,seed in enumerate(seeds):
        print("Num de seed:",i,"/",len(seeds))
        #Conversion chain
        value = seed
        for map in maps:
            value = convertFromMap(value,map)
        print("Final:",value)
        locations.append(value)
    
    print(min(locations))

def getMinconvertFromMap(start,range,map):
    """ int, list[list] -> Any"""
    for conversion_line in map:
        dest_start, origin_start, step = conversion_line
        end = start + range - 1 # Ejemplo 5,3 -> 5,6,7 Start:5 End:7 = 5+3-1

        #If the hole range is contained there the minimum is the dest start
        if origin_start <= start and end< origin_start + step:
            return dest_start

        #If the range is partially contained in the start and not in the end
        elif origin_start <= start and end >= origin_start + step:
            first_unconvertible = origin_start + step
            return min(dest_start,first_unconvertible)
        
        #If the range is partially contained in the end and not in the start
        elif origin_start > start and end < origin_start + step:
            last_unconvertible = origin_start-1
            return min(dest_start,last_unconvertible)
        
        #If the range is not even partially contained the min one is just the start one
        else:
            return start
        
        

def get_converted_range(ranges,map):
    """ list[[start,end],], list[list] -> Any"""
    """ Ejemplo: map: [[50, 98, 2] , [52, 50, 48]]
        range [[79,92]] -> range [[81,94]]
        range [[90,100]] -> range [[92,99],[50,51],[100,100]]"""
    new_ranges = []
    for i in range(len(ranges)):
        #Guarda el actual converting range para pasarlo por las distintas conversiones
        converting_range = ranges[i]

        #Por cada linea de conversión
        for conversion_line in map:
            #Si aún queda rango para transformar
            if len(converting_range) > 0:
                start,end = converting_range
                dest_start, origin_start, step = conversion_line
                offset = dest_start - origin_start

                #Si el start esta dentro del rango
                if origin_start <= start < origin_start + step:
                    #Si el fin también esta dentro del rango
                    if end < origin_start + step:
                        new_ranges.append([start+offset,end+offset])
                        converting_range = []
                    #Si el fin no está dentro del rango
                    else:
                        new_ranges.append([start+offset, origin_start + offset + step - 1])
                        converting_range[0] = origin_start + step
                
                #Si el start no está dentro del rango
                else:
                    #Si el fin esta dentro del rango:
                    if origin_start <= end < origin_start + step:
                        new_ranges.append([origin_start+offset,end+offset])
                        converting_range[1] = origin_start -1
                    #Si el fin no está dentro del rango
                    else:
                        #Si el rango está dentro del evaluado
                        if start < origin_start < end:
                            ranges.insert(i+1,[start,origin_start-1])
                            new_ranges.append([origin_start+offset,origin_start+offset+step-1])
                            ranges.insert(i+2,[origin_start+step,end])
                            converting_range = []
                        #Si el rango no esta dentro del evaluado esta fuera y se queda igual
                        else:
                            pass
        
        #Si no se ha conseguido convertir el rango, se mantiene igual
        if len(converting_range) > 0:
            new_ranges.append(converting_range)
        
    #Finalmente se retorna la nueva lista de rangos
    return new_ranges
        
                

    


def main2():
    lines = read_input("5",False)

    seed_ranges = [int(x) for x in lines[0].split(": ")[1].split(" ")]
    maps = get_maps(lines)

    min_locations = []
    for i in range(0,len(seed_ranges),2):
        print("Num de seed:",i,"/",len(seed_ranges))
        #Conversion chain
        ranges = [[seed_ranges[i],seed_ranges[i]+seed_ranges[i+1]-1]]
        for map in maps:
            ranges = get_converted_range(ranges,map)

        minimo = None
        print("Mínimo:",ranges)
        for pair in ranges:
            minimo_pair = min(pair)
            if minimo == None or minimo_pair < minimo:
                minimo = minimo_pair
        min_locations.append(minimo)
    
    print(min(min_locations))

#Calling
start_time = time.time()
main2()
end_time = time.time()
print(end_time-start_time)
