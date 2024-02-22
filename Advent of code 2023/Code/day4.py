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

def show_grid(grid,separator=""):
    for i, line in enumerate(grid): 
        print("{:<3}".format(i) + ": ", end = "")
        for col in line:
            print(col,end=separator)
        print()

def parse_games(lines):
    games = []
    for id,line in enumerate(lines):
        winning, numbers = str(line).split(" | ")
        winning = winning.split(": ")[1]
        winning = winning.split(" ")
        numbers = numbers.split(" ")
        game_winning_nums = []
        game_numbers = []
        for win_num in winning:
            if win_num != "":
                game_winning_nums.append(int(win_num))
        for your_num in numbers:
            if your_num != "":
                game_numbers.append(int(your_num))
            
        
        games.append([id,game_winning_nums,game_numbers,1])
    return games

def main():
    lines = read_input("4",False)
    games = parse_games(lines)
    game_points = []
    for i,game in enumerate(games):
        winning_nums = game[1]
        my_nums = game[2]
        points = 0
        for winning_num in winning_nums:
            if winning_num in my_nums:
                if points == 0:
                    points = 1
                else:
                    points*=2
        game_points.append(points)
    
    print(sum(game_points))

def get_num_matchs(game):
    win_nums = game[1]
    my_nums = game[2]
    counter = 0
    for num in win_nums:
        if num in my_nums:
            counter+=1
    return counter

def main2():
    lines = read_input("4",False)
    games = parse_games(lines) # [id, win_nums, your_nums,card_num]
    
    for i in range(len(lines)):
        num_matches = get_num_matchs(games[i])
        for x in range(num_matches):
            games[i+x+1][3] += (1*games[i][3])
    
    card_sum = 0
    for i,game in enumerate(games):
        print("Game",i+1,":",game[3])
        card_sum += game[3]
    print(card_sum)


     
#Calling
main2()
