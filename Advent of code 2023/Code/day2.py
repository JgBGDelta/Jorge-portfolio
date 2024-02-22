def read_input(num,is_example=False):
    lines = []
    path = "C:/Users/jorge/Desktop/UNIVERSIDAD/!Proyectos/Python/Advent of code 2023/Inputs/input" + num + ".txt"
    if is_example:
        path = "C:/Users/jorge/Desktop/UNIVERSIDAD/!Proyectos/Python/Advent of code 2023/Inputs/example" + num + ".txt"
    with open(path, 'r') as file:
        for line in file:
            lines.append(line.removeprefix("\n").removesuffix("\n"))
    return lines    

def show_data(game_results):
    for i,game_data in enumerate(game_results):
        print("Game " + str(i+1) + ": ",game_data)

def parse_game_results(lines):
    games_results = []
    #Parsing
    for line in lines:
        results_str = str(line).split(": ")[1]
        grab_results = results_str.split("; ")
        results = []
        for grab_result in grab_results:
            final_results_text = grab_result.split(", ")
            final_results = []
            for final_result in final_results_text:
                num, color = final_result.split(" ")
                final_results.append((int(num),color))                
            results.append(final_results)
        games_results.append(results)
    return games_results

def function():
    lines = read_input("2",False)
    games_results = parse_game_results(lines)

    configuration = {"red":12, "green":13, "blue":14}
    impossible_games_ids = []
    possible_game_ids = []
    for game_id, game in enumerate(games_results):
        for grab in game:
            for pair in grab:
                if pair[0] > configuration[pair[1]]:
                    impossible_games_ids.append(game_id+1)
                    break
            if game_id+1 in impossible_games_ids:
                break
        if not game_id+1 in impossible_games_ids:
            possible_game_ids.append(game_id+1)

    show_data(games_results)
    print(impossible_games_ids)
    print(possible_game_ids)
    print("Sum:", sum(possible_game_ids))

def function2():
    lines = read_input("2",False)
    games_results = parse_game_results(lines)

    powers = []
    for game in games_results:
        maximums = {'red':0,'blue':0,'green':0}
        for grab in game:
            for pair in grab:
                if pair[0] > maximums[pair[1]]:
                    maximums[pair[1]] = pair[0]
        power = 1
        for key in maximums:
            power *= maximums[key]
        powers.append(power)
    

    show_data(games_results)
    print("Sum powers:",sum(powers))



#Calling
function2()