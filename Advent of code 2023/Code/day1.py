def read_input(num,is_example=False):
    lines = []
    path = "C:/Users/jorge/Desktop/UNIVERSIDAD/!Proyectos/Python/Advent of code 2023/Inputs/input" + num + ".txt"
    if is_example:
        path = "C:/Users/jorge/Desktop/UNIVERSIDAD/!Proyectos/Python/Advent of code 2023/Inputs/example" + num + ".txt"
    with open(path, 'r') as file:
        for line in file:
            lines.append(line.removeprefix("\n").removesuffix("\n"))
    return lines    

def function():
    lines = read_input("1")
    nums = []
    nums_text = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9}
    for line in lines:
        
        first = -1
        current_line = ""
        #Pasada delantera
        for char in line:
            current_line += char
            if first != -1:
                break
            if str(char).isdigit():
                if first == -1:
                    first = int(char)
            for key in nums_text:
                if key in current_line:
                    first = nums_text[key]
        
        last = -1
        current_line = ""
        #Pasada trasera
        for i in range(len(line)-1,-1,-1):
            current_line = line[i] + current_line
            if last != -1:
                break
            if str(line[i]).isdigit():
                last = int(line[i])
            for key in nums_text:
                if key in current_line:
                    last = nums_text[key]
        
        nums.append(first*10 + last)
    
    print(sum(nums))
    
                    



#Calling
function()
