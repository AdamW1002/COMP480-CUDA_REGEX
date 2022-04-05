#Adam Weiss 3/16/2022 converts a regex to NFA


from ast import Expression
from cmath import exp
from enum import auto
from imp import init_builtin
from mimetypes import init
import re


alphabet = "qwertyuiopasdfghjklzxcvbnm" + "qwertyuiopasdfghjklzxcvbnm".upper() + "123456789"  + ",.?! "
class Node:
    count = 0
    node_list = []
    def __init__(self) -> None:
        
        self.transitions = dict() #where to go
        self.id = Node.count
        self.repeat = False  #should we self loop
        Node.count += 1
        Node.node_list += [self]


def find_parenthesis(search : str) -> int:
    depth = 0 #assume we start at depth 0
    for i  in range(len(search)):
        c = search[i]
        if c == "(": #seeing an opening makes it deeper
            depth += 1
        elif c == ")":
            depth -= 1

        if depth == 0:
            return i

def regex_to_NFA(expression : str) -> tuple:
    #returns a tuple, first entry is list of entry points, second entry is list of exit points
    if  not any ([ x in expression for x in "*+()|"]):
        initial = Node()
        #if len(expression) == 1: #if one letter then self loop
        #    initial.transitions[expression[0]] = [initial.id] #self loop and end
        #    return initial, initial 
        if len(expression) == 1:
            current = Node()
            initial.transitions[expression[0]] = [current.id]
            return initial,current
        else:
            current = initial #move along letters connecting but self loop at the end
            for c in expression:
                new_node = Node()
                current.transitions[c] = [new_node.id]
                current = new_node
            #current.transitions[expression[-1]] = current.id
        return (initial, current)
    elif "|" in expression and not any ([ x in expression for x in "()"]):
        units = expression.split("|") # break down each path
        paths = [regex_to_NFA(x) for x in units ]
        initial = Node()
        final = Node()
        for head,tail in paths: #  say paths are ABC and DEF then initial -> ABC and initial ->DEF
            join(initial, head)
            join(tail, final)
        #        /> path 2 \>
        #initial -> path 1 -> final
        #        \> path 0 />
        return (initial, final)

    elif "*" in expression and not any ([ x in expression for x in "()"]):
        units = expression.split("*")#abc*def*gh -> abc def gh
        repeat_chars = [x[-1] for x in units[:-1]] #last char is repeated abc def gh -> c f h
        non_repeat_chars = [x[:-1] for x in units[:-1]] #abc def gh -> ab de g 

        initial = Node()
        final = Node()
        current = initial
        for i in range (len(repeat_chars)):      
            repeat = repeat_chars[i]
            non_repeat = non_repeat_chars[i]

            repeat_start = Node() #outer 
            repeat_end = Node()

            internal_start,internal_end = regex_to_NFA(repeat)
            internal_start.transitions[repeat] = [internal_start] #self loop

            if non_repeat != "": #if something to repeat
                non_repeater_begin, non_repeater_end = regex_to_NFA(non_repeat)
                join(current, non_repeater_begin)
                join(non_repeater_end, repeat_start)

            else:
                join(current,repeat_start)

            current = repeat_end
        
        return (initial, current)
    else:
        matching_paren = find_parenthesis(expression)
        block1 = expression[:matching_paren] #get first parenthesis part
        block1_regex = expression[1:matching_paren]
        block1_start, block1_end = regex_to_NFA(block1_regex)
        if expression[matching_paren:] == "": #if whole thing in parentheses
            return (block1_start,block1_end) #regex to nfa part with no parens
        else:
            block2_regex = expression[matching_paren+1:] #recursively decompose second block
            block2_start, block2_end = regex_to_NFA(block2_regex)
           
            join(block1_end, block2_start) #stick second block onto first
           
            return (block1_start, block2_end)
            

def printNFA(NFA : Node, visited : set) -> None:
    if  not NFA in visited:
        visited.add(NFA)
        print("At node {} with transitions {}".format(NFA.id, NFA.transitions))
        for transitions in NFA.transitions.values():
            for transition in transitions:
                printNFA(NFA.node_list[transition],visited)


def join(host : Node, graft : Node) -> Node:
    for symbol in alphabet: #basically an epsilon transition
        
        if symbol in host.transitions:
            
            host.transitions[symbol] += [graft.id]
        else:
            host.transitions[symbol] = [graft.id]
    


def regexToTable(NFA : Node, acc : dict) -> dict:
    if NFA.id not in acc.keys(): #if not been here before
        transition_string = ""
        for key,value in NFA.transitions.items():
            transition_string += "\t{}\t{}".format(key,value)
        #print(transition_string)

        acc[NFA.id] = transition_string

        for transitions in NFA.transitions.values():
            for transition in transitions:
                regexToTable(NFA.node_list[transition], acc)
def TableToString(table : dict):
    acc = ""
    for k in table.keys():
        acc += str(k) + table[k] + "\n"
    return acc
#n1 = Node()
#n2 = Node()
#
#print(n1.count)
#print(n2.count)
#print(n1.id)
#print(n2.id)
#print(n1.node_list)
#print(n2.node_list)

#expression = "abc"
#head1,tail1= regex_to_NFA(expression)
#print(head1.transitions)
#printNFA(head1, set())
#
#expression2 = "def"
#
#head2,tail2 = regex_to_NFA(expression)
#print(head2.transitions)
#printNFA(head2, set())
#
#join(tail1,head2)
#printNFA(head1,set())

#expression3 = "abc|def|ghi"
#head3, tail3 = regex_to_NFA(expression3)
#printNFA(head3, set())

#expression4 = "c"
#head4,tail4 = regex_to_NFA(expression4)
#printNFA(head4, set())
s = "(fsdafsd(fdsfsa)fdsfsa)fsad(sdf)"
print(s[8+7])
print(find_parenthesis(s[8:]), s[find_parenthesis(s[8:])])

expr5 = "(a|b)(cd)"
head5,tail5 = regex_to_NFA(expr5)
printNFA(head5, set())

print("max node {}".format( max(Node.node_list, key=lambda x : x.id).id))
test_dict = dict()
print(regexToTable(head5,test_dict))
print(TableToString(test_dict))
f = open("nfa.txt", "w")
f.write((TableToString(test_dict)))