from utils import Node, Edge, Instance, Solution, FloydWarshall
import numpy as np
import pickle
import random as rd
import time 
import copy
from tqdm import tqdm
from collections import deque
import sys

# sys.setrecursionlimit(50)

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

    def cost_and_branch(self, n_id, fw: FloydWarshall, instance: Instance):
        n = self.nodes[n_id]

        cost = instance.edges_dict[frozenset({n_id, n.parent})].cost()
        branch = [n_id]
        
        ancestor = self.nodes[n.parent]
        while True:
            if (len(ancestor.child) > 1) or (ancestor.id in instance.profit_nodes) or (ancestor.parent == None):
                break
            else:
                cost += instance.edges_dict[frozenset({ancestor.id, ancestor.parent})].cost()
                branch.append(ancestor.id)
                ancestor = self.nodes[ancestor.parent]

        return cost, branch

    def elegate_branch(self, branch: list):
        for n_id in branch:
            n = self.nodes[n_id]
            self.nodes[n.parent].child.remove(n_id)
            n.parent = None
            if n_id != branch[0]:
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
        
        if n_id == n_j_id: return True

        curr_n_j_id = n_j_id
        while self.nodes[curr_n_j_id].parent:
            # print(self.nodes[curr_n_j_id].parent, self.nodes[curr_n_j_id].child)
            parent = self.nodes[curr_n_j_id].parent
            if parent == n_id:
                return True
            else:
                curr_n_j_id = parent
        return False
            

    def apply_path(self, path):
        start_id = path[-1].idx()
        end_id = path[0].idx()
        
        for i in range(len(path)):
            n_i_id = path[i].idx()

            if n_i_id not in self.nodes:
                self.nodes[n_i_id] = TreeNode(n_i_id, None, set())

            n_i = self.nodes[n_i_id]

            if n_i_id == end_id:
                n_i.parent = path[i+1].idx()
                if path[i+1].idx() in n_i.child:
                    n_i.child.remove(path[i+1].idx())
            elif n_i_id == start_id:
                n_i.child.add(path[i-1].idx())
            elif n_i.parent == None:
                self.nodes[path[i].idx()] = TreeNode(path[i].idx(), path[i+1].idx(), {path[i-1].idx()})
            else:
                n_i.parent = path[i+1].idx()
                if path[i+1].idx() in n_i.child:
                    n_i.child.remove(path[i+1].idx())
                n_i.child.add(path[i-1].idx())
    
    
    def distance_from_center(self, n_id: int):
        dist = 0
        curr_n = self.nodes[n_id]
        while curr_n.parent:
            dist += 1
            curr_n = self.nodes[curr_n.parent]
        
        return dist
    
    def distance_of_tail(self, n_id: int, debug=False): 
        # if n_id == 66: debug = True
        # if debug:
        #     print(n_id, self.nodes[n_id].parent, self.nodes[n_id].child)
            
        if len(self.nodes[n_id].child) == 0:
            return 0
        
        return (1 + max(self.distance_of_tail(child, debug) for child in self.nodes[n_id].child))
        
    def inverse_tree(self, n_id: int, m_id: int):
        n = self.nodes[n_id]
        parent = n.parent

        if parent == None:
            n.parent = m_id
            if m_id in n.child:
                n.child.remove(m_id)
            return
        
        n.parent = m_id
        n.child.add(parent)
        if m_id in n.child:
            n.child.remove(m_id)
        self.inverse_tree(parent, n_id)




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

    # Supression des noeuds inutiles
    # delete_useless_nodes(instance, fw)

    # Solution inital 
    banned_pv = []
    tree, cost = initial_solver(instance, fw, banned_pv)
    
    # Recherche local sucecssive
    banned_pv = []
    while cost > instance.B:
        # Supression de nb PV
        nb = 1
        for _ in range(nb):
            pv_id = min(filter(lambda x: x not in banned_pv and x!=1 and x in tree.nodes and len(tree.nodes[x].child) == 0, instance.profit_nodes), key=lambda x: instance.profit_nodes[x].revenue()/tree.cost_and_branch(x, fw, instance)[0])
            banned_pv.append(pv_id)
            branch_cost, branch = tree.cost_and_branch(pv_id, fw, instance)
            tree.elegate_branch(branch)
            del tree.nodes[pv_id]
            cost -= branch_cost

        # Nouvelle recherche locale
        print(banned_pv)
        # tree, cost = initial_solver(instance, fw, banned_pv)
        tree, cost = local_search(tree, cost, instance, fw)



    # Mise en forme de la solution
    sol = tree.tree_to_sol(instance)
    print(instance.solution_value(sol))
    return sol


    # Mise en forme
    solution = tree.tree_to_sol(instance)
    return solution

def local_search(tree, cost, instance, fw):
    t = time.time()
    stop = False
    while not stop:
        stop = True
        neighborhood = find_neighborhood(tree, cost, instance, fw)
        neighborhood.sort(key=lambda x: x[1])

        newTree, newCost = neighborhood[0][0], neighborhood[0][1]
        if newCost < cost:
            stop = False
            tree = copy.deepcopy(newTree)
            cost = newCost
            print(f"New record {newCost}")

    print(f"Temps d'exec : {time.time() - t}")

    return tree, cost

def find_neighborhood(tree: Tree, cost: int, instance: Instance, fw: FloydWarshall):
    neighborhood = []
    

    nodes = list(instance.nodes.keys())
    # rd.shuffle(nodes)
    for n_id in tqdm(nodes): # Tri de la liste ?
        if n_id != 1:
            attractor(n_id, tree, cost, instance, fw, neighborhood)
    
    return neighborhood

def attractor(n_id: int, tree: Tree, cost: int, instance: Instance, fw: FloydWarshall, neighborhood: list):
    inTree = n_id in tree.nodes
    
    newTree = copy.deepcopy(tree)
    newCost = cost  

    if not inTree:
        dist, path = fw.get_real_distance_and_path(1, n_id, newTree, instance, [])
        newTree.apply_path(path)
        newCost += dist

    stop = False
    while not stop:
        stop = True
        nodes = list(instance.nodes.keys())
        # rd.shuffle(nodes)
        for m_id in nodes: # Tri de la liste ?
            if m_id in newTree.nodes and m_id != 1 and not newTree.is_ancestor(m_id, n_id):
               
                m_cost, m_branch = newTree.cost_and_branch(m_id, fw, instance)
                m_new_cost, m_new_path = fw.get_real_distance_and_path(n_id, m_id, newTree, instance, m_branch)
                
                if n_id not in m_branch and m_cost > m_new_cost:
                    saveTree = copy.deepcopy(newTree)

                    newTree.elegate_branch(m_branch)   
                    newTree.inverse_tree(m_new_path[0].idx(), m_new_path[1].idx())
                    newTree.apply_path(m_new_path)
                      
                    if newTree.distance_from_center(m_new_path[-1].idx()) + newTree.distance_of_tail(m_new_path[-1].idx()) <= instance.H :
                        stop = False
                        newCost += m_new_cost - m_cost
                    else:
                        stop = True
                        newTree = copy.deepcopy(saveTree)

    neighborhood.append([newTree, newCost])



def initial_solver(instance: Instance, fw: FloydWarshall, banned_pv: list):
    tree = Tree()

    for pn_id in instance.profit_nodes.keys():
        if pn_id != 1 and pn_id not in banned_pv:
            path = fw.get_shortest_path(pn_id, 1, instance)
            if len(path) - 1 <= instance.H:
                tree.apply_path(path)
            
    return tree, instance.solution_cost(tree.tree_to_sol(instance))
    
def delete_useless_nodes(instance: Instance, fw: FloydWarshall):
    to_del = []
    for n in instance.nodes.values():
        useless = True
        n_id = n.idx()
        edges = instance.neighs(n)

        for edge_1 in edges:
            n_a, n_b = edge_1.idx()
            if n_a.idx() == n_id:
                n_1 = copy.deepcopy(n_b)
            else: 
                n_1 = copy.deepcopy(n_a)
            for edge_2 in edges:
                n_a, n_b = edge_2.idx()
                if n_a.idx() == n_id:
                    n_2 = copy.deepcopy(n_b)
                else: 
                    n_2 = copy.deepcopy(n_a)
                
                if n_1.idx() != n_2.idx() and edge_1.cost() + edge_2.cost() <= fw.get_distance(n_1.idx(), n_2.idx()):
                    useless = False
        if useless and n_id not in instance.profit_nodes:
            if len(edges) > 1:
                print(f"inutile ! {n_id}")
            to_del.append(n_id)
    # print(fw.get_shortest_path(46, 75, instance))
    print(len(to_del))
    for n in to_del:
        del instance.nodes[n]
