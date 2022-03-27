#Adam Weiss 3/16/2022 converts a regex to NFA


from cmath import exp
from enum import auto
from imp import init_builtin
from mimetypes import init
import re


alphabet = "qwertyuiopasdfghjklzxcvbnm" + "qwertyuiopasdfghjklzxcvbnm".upper() + "123456789"
class Node:
    count = 0
    node_list = []
    def __init__(self) -> None:
        
        self.transitions = dict() #where to go
        self.id = Node.count
        self.repeat = False  #should we self loop
        Node.count += 1
        Node.node_list += [self]

def regex_to_NFA(expression : str) -> tuple:
    #returns a tuple, first entry is list of entry points, second entry is list of exit points
    if  not any ([ x in expression for x in "*+()|"]):
        initial = Node()
        if len(expression) == 1: #if one letter then self loop
            initial.transitions[expression[0]] = [initial.id] #self loop and end
            return initial, initial 
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
        for head,tail,i in paths: #  say paths are ABC and DEF then initial -> ABC and initial ->DEF
            join(initial, head)
            join(tail, final)
        #        /> path 2 \>
        #initial -> path 1 -> final
        #        \> path 0 />
        

    elif "*" in expression and not any ([ x in expression for x in "()"]):
        pass




def printNFA(NFA : Node, visited : set) -> None:
    if NFA.transitions != dict() and not NFA in visited:
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

expression3 = "abc|def|ghi"
head3, tail3 = regex_to_NFA(expression3)
printNFA(head3, set())