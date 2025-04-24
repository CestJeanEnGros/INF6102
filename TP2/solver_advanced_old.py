from utils import Node, Edge, Instance, Solution, FloydWarshall
import numpy as np
import pickle
import time 
from tqdm import tqdm
from collections import deque

class CustomNode(Node):

    """ You are completely free to extend classes defined in utils,
        this might prove useful or enhance code readability
    """

    def __init__(self, idx, revenue):
        super().__init__(idx, revenue)

class TreeNode():
    def __init__(self, id, parent=None, child=set()):
        self.id = id
        self.parent = parent
        self.child = child

class Tree():
    def __init__(self):
        self.nodes = {1: TreeNode(1)}

    def cost_of_branch(self, n_id, fw: FloydWarshall, instance: Instance):
        branch_ids = [n_id]
        n = self.nodes[n_id]
        ancestor = self.nodes[n.parent]
        while True:
            if (len(ancestor.child) > 1) or (ancestor.id in instance.profit_nodes) or (ancestor.parent == None):
                break
            else:
                branch_ids.append(ancestor.id)
                ancestor = self.nodes[ancestor.parent]

        return fw.get_distance(n_id, ancestor.id), branch_ids

    def elegate_branch(self, branch_ids: list):
        for n_id in branch_ids:
            n = self.nodes[n_id]
            self.nodes[n.parent].child.remove(n_id)
            n.parent = None
            if n_id != branch_ids[0]:
                del self.nodes[n_id]
            

    def find_best_jonction(self, n_id, branch_cost, branch_ids, fw: FloydWarshall):
        best_cost = branch_cost
        best_id = None
        for n_j_id in self.nodes.keys():
            if not self.is_ancestor(n_id, n_j_id):
                cost = fw.get_distance(n_id, n_j_id)
                if n_j_id not in branch_ids and cost < best_cost:
                    best_cost = cost
                    best_id = n_j_id
        return best_id

    def is_ancestor(self, n_id, n_j_id):
        curr_n_j_id = n_j_id
        while self.nodes[curr_n_j_id].parent:
            parent = self.nodes[curr_n_j_id].parent
            if parent == n_id:
                return True
            else:
                curr_n_j_id = parent
        return False
            

    def apply_jonction(self, n_id, n_j_id, path):
        n = self.nodes[n_id]
        n_j = self.nodes[n_j_id]
        for i in range(len(path)):
            n_p_id = path[i].idx()
            if n_p_id == n_id:
                n.parent = path[i+1].idx()
            elif n_p_id == n_j_id:
                n_j.child.add(path[i-1].idx())
            else:
                if path[i].idx() in self.nodes:
                    if path[i+1].idx() in self.nodes[path[i].idx()].child:
                        self.nodes[path[i].idx()].child.remove(path[i+1].idx())
                    self.nodes[path[i].idx()].parent = path[i+1].idx()
                    self.nodes[path[i].idx()].child.add(path[i-1].idx())
                else:
                    self.nodes[path[i].idx()] = TreeNode(path[i].idx(), path[i+1].idx(), {path[i-1].idx()})
                

        
        
    
    def tree_to_sol(self, instance: Instance):
        path = []
        edgesToAdd = []
        for n in self.nodes.values():
            if n.parent:
                edgesToAdd.append(Edge(Node(n.id), Node(n.parent)))
        
        for edge in instance.edges:
            if edge in edgesToAdd:
                path.append(edge)
        
        return Solution(set(path))
        

def solve(instance: Instance) -> Solution:
    """Write your code here

    Args:
        instance (Instance): An Instance object containing all you need to solve the problem

    Returns:
        Solution: A solution object initialized with an iterator on Edge object
    """
    # Setup
    if True:
        if instance.B == 85:
            with open("instanceA.pkl", "rb") as fichier:
                fw = pickle.load(fichier)
        elif instance.B == 103:
            with open("instanceB.pkl", "rb") as fichier:
                fw = pickle.load(fichier)
        elif instance.B == 342:
            with open("instanceC.pkl", "rb") as fichier:
                fw = pickle.load(fichier)
        elif instance.B == 272:
            with open("instanceD.pkl", "rb") as fichier:
                fw = pickle.load(fichier)
        elif instance.B == 136:
            with open("instanceE.pkl", "rb") as fichier:
                fw = pickle.load(fichier)
        elif instance.B == 345:
            with open("instanceF.pkl", "rb") as fichier:
                fw = pickle.load(fichier)
    else:
        startTime = time.time()
        fw = FloydWarshall(instance)
        fw.floyd_warshall(instance)
        print(f"Temps d'éxécution du setup : {time.time() - startTime}")

    # Solution inital 
    tree = initial_solver(instance, fw)
    
    # Amélioration
    stop = False
    while not stop:
        stop = True
        for n_id in instance.nodes:
            if n_id in tree.nodes.keys() and n_id != 1:
                cost, branch = tree.cost_of_branch(n_id, fw, instance)
                n_j_id = tree.find_best_jonction(n_id, cost, branch, fw)
                if n_j_id != None:
                    stop = False
                    tree.elegate_branch(branch)
                    tree.apply_jonction(n_id, n_j_id, fw.get_shortest_path(n_id, n_j_id, instance))
    
    # Rectification du budget
    sortedProfitsNodes = deque(sorted(filter(lambda x: x!=1, instance.profit_nodes), key=lambda x: instance.profit_nodes[x].revenue()/tree.cost_of_branch(x, fw, instance)[0]))
    print(sortedProfitsNodes)
    solution = tree.tree_to_sol(instance)
    totalCost = instance.solution_cost(solution)

    while totalCost > instance.B:
        n_id = sortedProfitsNodes.popleft()
        if len(tree.nodes[n_id].child) == 0:
            cost, branch = tree.cost_of_branch(n_id, fw, instance)
            tree.elegate_branch(branch)
            totalCost -= cost
            print(n_id, cost, branch)

    # Mise en forme
    solution = tree.tree_to_sol(instance)
    return solution

    

def initial_solver(instance: Instance, fw: FloydWarshall):
    tree = Tree()

    sortedProfitsNodes = sorted(filter(lambda x: x!=1, instance.profit_nodes), key=lambda x: instance.profit_nodes[x].revenue()/fw.get_distance(1, x), reverse=True)
    
    path = []
    cost = 0

    for pn_id in sortedProfitsNodes:
        subPathNodes = fw.get_shortest_path(1, pn_id, instance)

        for i in range(len(subPathNodes)):
            n_id = subPathNodes[i].idx()
            if n_id == 1:
                n_s_id = subPathNodes[i+1].idx()
                if n_id in tree.nodes:
                    tree.nodes[n_id].child.add(n_s_id)
                else:
                    tree.nodes[n_id] = TreeNode(n_id, None, {n_s_id})
            elif n_id == pn_id:
                n_p_id = subPathNodes[i-1].idx()
                if n_id in tree.nodes:
                    tree.nodes[n_id].parent = n_p_id
                else:
                    tree.nodes[n_id] = TreeNode(n_id, n_p_id, set())
            else:
                n_s_id = subPathNodes[i+1].idx()
                n_p_id = subPathNodes[i-1].idx()
                if n_id in tree.nodes:
                    tree.nodes[n_id].parent = n_p_id
                    tree.nodes[n_id].child.add(n_s_id)          
                else:
                    tree.nodes[n_id] = TreeNode(n_id, n_p_id, {n_s_id})
    return tree
    


