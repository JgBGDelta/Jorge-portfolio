
def read_input(num,is_example=False):
    lines = []
    path = "C:/Users/jorge/Desktop/UNIVERSIDAD/!Proyectos/Python/Advent of code 2023/Inputs/input" + num + ".txt"
    if is_example:
        path = "C:/Users/jorge/Desktop/UNIVERSIDAD/!Proyectos/Python/Advent of code 2023/Inputs/example" + num + ".txt"
    with open(path, 'r') as file:
        for line in file:
            lines.append(line.removeprefix("\n").removesuffix("\n"))
    return lines    

def is_valid(chain,nums,init_index = 0,init_pos = 0):
    current_count = 0
    
    index = init_index
    validated_index = init_pos
    
    for k in range(init_pos,len(chain)+1):
        if k != len(chain) and chain[k] == '#':
            current_count += 1
        elif k == len(chain) or chain[k] == '.':
            if current_count != 0 and (index >= len(nums) or current_count != nums[index]): 
                return False,index,validated_index
            else:
                if current_count != 0:
                    index += 1
                    validated_index = k
                current_count = 0
        else:
            if (not index >= len(nums)) and current_count > nums[index]:
                return False,index,validated_index
            else:
                return True,index,validated_index
        
    if index < len(nums):
        return False,index,validated_index
    else:
        return True,index,validated_index
    

def get_combination(chain,chars,nums,maxHash,hashCount,index=0,validated_index=0):
    if hashCount > maxHash:
        return 0
    if not '?' in chain:
        valid,index,validated_index= is_valid(chain,nums,index,validated_index)
        return 1 if valid else 0
    else:
        total = 0
        valid,index,validated_index = is_valid(chain,nums,index,validated_index)
        if not valid:
            return 0
        for char in chars:
            extra = 0
            if char == '#':
                extra = 1
            total += get_combination(str(chain).replace('?',char,1),chars,nums,maxHash,hashCount+extra,index,validated_index)
                    
        return total



def main():
    lines = read_input("12",True)
    
    total = 0
    for h,line in enumerate(lines):
        seq,nums = line.split(' ')
        nums = [int(x) for x in nums.split(',')]
        
        #Part 2
        part2 = True
        if(part2):
            seq = seq + ('?'+seq)*4
            nums_copy = nums[:]
            for k in range(4):
                nums.extend(nums_copy)
            print(h+1,'/',len(lines),seq,nums)


        maxHash = sum(nums)
        hashCount = 0
        for char in seq:
            if char == '#':
                hashCount +=1
        #Determinar las posibilidades - Enfoque bruteforce
        combination_count = get_combination(seq,['#','.'],nums,maxHash,hashCount)
        total += combination_count
    
    print(total)

        



#Calling
main()
# print(get_combination('?.??#',['#','.'],[1]))
