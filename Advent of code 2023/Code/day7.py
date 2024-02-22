import random

def read_input(num,is_example=False):
    lines = []
    path = "C:/Users/jorge/Desktop/UNIVERSIDAD/!Proyectos/Python/Advent of code 2023/Inputs/input" + num + ".txt"
    if is_example:
        path = "C:/Users/jorge/Desktop/UNIVERSIDAD/!Proyectos/Python/Advent of code 2023/Inputs/example" + num + ".txt"
    with open(path, 'r') as file:
        for line in file:
            lines.append(line.removeprefix("\n").removesuffix("\n"))
    return lines    

def get_score(hand):
    card_count = {}
    j_count = 0
    for card in hand:
        if card != "J":
            if card in card_count:
                card_count[card] +=1
            else:
                card_count[card] = 1
        else:
            j_count += 1
    
    if JOKER and j_count >0:
        max_key,max_count = '',0
        for key in card_count:
            if card_count[key] > max_count:
                max_count = card_count[key]
                max_key = key
        if max_key == '':
            return 7
        else:
            card_count[max_key] += j_count

    type_ranking = 1 #7 All of a kind, 6 Four of a kind, 5 Full house, 4 Three of a kind, 3 Two pair, 2 One pair
    for key in card_count:
        if card_count[key] == 5:
            type_ranking = 7
            break
        elif card_count[key] == 4:
            type_ranking = 6
            break
        elif card_count[key] == 3:
            if type_ranking == 2:
                type_ranking = 5
                break
            else:
                type_ranking = 4
        elif card_count[key] == 2:
            if type_ranking == 4:
                type_ranking = 5
                break
            elif type_ranking == 2:
                type_ranking = 3
                break
            else:
                type_ranking = 2
    return type_ranking

def insert_in_place(hand,score,bid,ranked_list,end=None,start=0):
    if end == None:
        end = len(ranked_list)
    
    while start < end:
        mid = (start+end)//2
        if is_better((hand,bid,score),ranked_list[mid]):
            start = mid + 1
        else:
            end = mid
    ranked_list.insert(start,(hand,bid, score))


def is_better(comb1,comb2):
    hand1,bid1,type_ranking1 = comb1
    hand2,bid2,type_ranking2 = comb2
    if not JOKER:
        values = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'T':10,'J':11,'Q':12,'K':13,'A':14}
    else:
        values = {'J':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'T':10,'Q':12,'K':13,'A':14}
    if type_ranking1 > type_ranking2:
        return True
    elif type_ranking1 <type_ranking2:
        return False
    else:
        for i in range(len(hand1)):
            if values[hand1[i]] > values[hand2[i]]:
                return True
            elif values[hand1[i]] < values[hand2[i]]:
                return False
    return False

    

def main():
    lines = read_input("7",False)
    ranked = []
    for line in lines:
        hand,bid = line.split(" ")
        bid = int(bid)
        hand_Score = get_score(hand)
        insert_in_place(hand,hand_Score,bid,ranked)

    total = 0
    for i, pair in enumerate(ranked):
        total += (i+1)*pair[1]
    
    print(ranked)
    print(total)

#Calling
JOKER = True
main()


