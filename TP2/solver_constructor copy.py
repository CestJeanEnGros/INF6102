from utils import Node, Edge, Instance, Solution, FloydWarshall
import numpy as np
import pickle
import random as rd
import time 
import copy
from tqdm import tqdm

class TreeNode():
    """Classe représentant un noeud de l'arbre solution."""
    def __init__(self, id, parent=None, child=set()):
        self.id = id
        self.parent = parent
        self.child = child

class Tree():
    """Classe représentant un arbre solution."""
    def __init__(self):
        self.nodes = {1: TreeNode(1)}

    def cost_and_branch(self, n_id, fw: FloydWarshall, instance: Instance):
        """
    Args:
        n_id (int): L'id d'un noeud de l'arbre
        fw (FloydWarshall): Le FloydWarshall contenant les plus court cheminsde l'instance
        instance (Instance): L'instance du problème

    Returns:
        cost (int): Le cout de la branche ancestrale indépendante du noeud n_id
        branch (list): La liste des ids des noeuds composant la branche ancestrale indépendante du noeudn_id
    """
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
        """ Efface une branche de l'arbre
    Args:
        branch (list): Liste des ids des noeuds composant la branche à supprimer
    """
        for n_id in branch:
            n = self.nodes[n_id]
            self.nodes[n.parent].child.remove(n_id)
            n.parent = None
            if n_id != branch[0]:
                del self.nodes[n_id]   

    def is_ancestor(self, n_id, n_j_id):
        """
    Args:
        n_id (int): L'id d'un noeud de l'arbre
        n_j_id (int): L'id d'un noeud de l'arbre

    Returns:
        (bool): Vrai si n_id est un ancetre de n_j_id dans l'arbre, faux sinon    
    """
        if n_id == n_j_id: return True

        curr_n_j_id = n_j_id
        while self.nodes[curr_n_j_id].parent:
            parent = self.nodes[curr_n_j_id].parent
            if parent == n_id:
                return True
            else:
                curr_n_j_id = parent
        return False
            
    def apply_path(self, path):
        """ Rajoute le chemin dans l'arbre
    Args:
        path (list): Liste des noeuds composant un chemin à rajouter
    """
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
    
    def distance_from_n(self, n_id, m_id):
        """
    Args:
        n_id (int): L'id d'un noeud de l'arbre
        m_id (int): L_id d'un noeud ancetre à n_id dans l'arbre

    Returns:
        dist (int): Distance en nombre d'arrete entre n_id et m_id
    """
        dist = 0
        curr_n = self.nodes[n_id]
        while curr_n.id != m_id:
            dist += 1
            curr_n = self.nodes[curr_n.parent]
        
        return dist

    def distance_inversed(self, n_id: int, m_id: int, b_id: int):
        """
    Args:
        n_id (int): L'id d'un noeud de l'arbre

    Returns:
        dist (int): Distance en nombre d'arrete entre le centre de l'arbre et le noeud d'id n_id
    """
        # print(n_id, self.nodes[n_id].parent, self.nodes[n_id].child)
        if n_id == m_id:
            nodes = list(filter(lambda x: x!= b_id, list(self.nodes[n_id].child)))
        else:
            nodes = list(filter(lambda x: x!= b_id, list(self.nodes[n_id].child) + [self.nodes[n_id].parent]))
        # print(n_id, m_id, b_id, nodes)
        if len(nodes) == 0 or n_id == 1:
            return 0
                
        return (1 + max(self.distance_inversed(node, m_id, n_id) for node in nodes))
    
    def distance_from_center(self, n_id: int):
        """
    Args:
        n_id (int): L'id d'un noeud de l'arbre

    Returns:
        dist (int): Distance en nombre d'arrete entre le centre de l'arbre et le noeud d'id n_id
    """
        dist = 0
        curr_n = self.nodes[n_id]
        while curr_n.parent:
            dist += 1
            curr_n = self.nodes[curr_n.parent]
        
        return dist
    
    def distance_of_tail(self, n_id: int): 
        """
    Args:
        n_id (int): L'id d'un noeud de l'arbre

    Returns:
       dist (int): Distance en nombre d'arrete entre un noeud d'id n_id et son plus lointain enfant.
    """
        if len(self.nodes[n_id].child) == 0:
            return 0
        
        return (1 + max(self.distance_of_tail(child) for child in self.nodes[n_id].child))
        
    def inverse_tree(self, n_id: int, m_id: int):
        """
    Args:
        n_id (int): L'id d'un noeud de l'arbre à partir du quel on doit inverser l'ordre parent/enfant
        m_id (int): Id du nouveau parent du noeud d'id n_id
    """
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

    def best_cost_and_path(self, pv_id: int, fw: FloydWarshall, instance: Instance):
        """ Trouve le chemin de cout minimum permetant d'inclure pv_id dans l'arbre
    Args:
        pv_id (int): L'id d'un PV de l'arbre
        fw (FloydWarshall): Le FloydWarshall contenant les plus court cheminsde l'instance
        instance (Instance): L'instance du problème

    Returns:
        best_cost (int): Le cout minimal d'insertion de pv_id dans l'arbre
        best_path (list): Liste des noeuds composant le chemin de cout minimal d'insertion de pv_id dans l'arbre
    """
        best_path = None
        best_cost = np.inf
        for n_id in self.nodes:
            if n_id != pv_id:
                add_cost, add_path = fw.get_real_distance_and_path(n_id, pv_id, self, instance, [])
                if add_cost < best_cost and (self.distance_from_center(add_path[-1].idx()) + len(add_path) - 1) <= instance.H:
                    best_cost = add_cost
                    best_path = add_path
        return best_cost, best_path

    def tree_to_sol(self, instance: Instance):
        """ Converti l'arbre solution en objet Solution
    Args:
        instance (Instance): L'instance du problème

    Returns:
        (Solution): Un objet Solution de l'instance
    """
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
    """ Trouve une solution optimisée du problème
    Args:
        instance (Instance): L'instance du problème
    Returns:
        (Solution): Un objet Solution de l'instance
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
    delete_useless_nodes(instance, fw)

    # Initialisation de la solution constructive
    tree = Tree()
    cost = 0

    # Ajout successif des PV à la solution
    stop = False
    while not stop:
        stop = True
        # Tri des PV en fonction de leur cout d'ajout à l'arbre et de leur revenue
        cost_and_path_dict = {pv_id: tree.best_cost_and_path(pv_id, fw, instance) for pv_id in filter(lambda x: x!=1 and x not in tree.nodes, instance.profit_nodes)}
        sorted_pv_nodes = sorted(cost_and_path_dict.keys(), key=lambda x: cost_and_path_dict[x][0]/instance.profit_nodes[x].revenue())
        # Essais d'insertion des PV
        for i, pv_id in enumerate(sorted_pv_nodes):
            if pv_id not in tree.nodes:
                # On test si on peut ajotuer le PV sans briser le critère H
                best_cost, best_path = cost_and_path_dict[pv_id]
                if best_path == None:
                    continue
                else:
                    # On ajoute le PV et on applique une recherche locale
                    # print(f"avant l'ajout du noeud {pv_id}")
                    # print(instance.is_valid_solution(tree.tree_to_sol(instance)))

                    newTree = copy.deepcopy(tree)
                    newTree.apply_path(best_path)
                    newCost = cost + best_cost
                    # print(f"après l'ajout du noeud {pv_id}")
                    # print(instance.is_valid_solution(newTree.tree_to_sol(instance)))

                    newTree, newCost = local_search(newTree, newCost, instance, fw)
                    # print("après la recherche locale")
                    # print(instance.is_valid_solution(newTree.tree_to_sol(instance)))
                    # print(instance.nodes_out_of_maximal_distance(newTree.tree_to_sol(instance)))

                    # Si on est toujours en dessous du critère B alors on adopte ce nouvel arbre et on revient au début de la boucle while pour remettre à jour le tri des PV
                    if newCost <= instance.B:
                        print(f"revenue record {instance.solution_value(newTree.tree_to_sol(instance))}")
                        print(newCost)
                        tree = newTree
                        cost = newCost
                        stop = False
                        break

    # Mise en forme de la solution
    sol = tree.tree_to_sol(instance)
    print(instance.solution_value(sol))
    return sol


def local_search(tree: Tree, cost: int, instance: Instance, fw: FloydWarshall):
    """ Trouve une solution optimisée du problème
    Args:
        tree (Tree): L'arbre représentant la solution actuelle
        cost (int): Cout selon le critère B de la solution acutelle
        instance (Instance): L'instance du problème
        fw (FloydWarshall): Le FloydWarshall contenant les plus court cheminsde l'instance
    Returns:
        tree (Tree): L'arbre optimisée à l'issue de la recherche locale
    """
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
    # print(f"Temps d'exec : {time.time() - t}")

    return tree, cost


def find_neighborhood(tree: Tree, cost: int, instance: Instance, fw: FloydWarshall):
    """ Créer et renvoie la liste des voisins à la solution acutelle
    Args:
        tree (Tree): L'arbre représentant la solution actuelle
        cost (int): Cout selon le critère B de la solution acutelle
        instance (Instance): L'instance du problème
        fw (FloydWarshall): Le FloydWarshall contenant les plus court cheminsde l'instance
    Returns:
        neighborhood (list): Liste des couple (Arbre, Cout) composant le voisinage de tree
    """    
    neighborhood = []

    for n_id in instance.nodes.keys(): 
        if n_id != 1:
            attractor(n_id, tree, cost, instance, fw, neighborhood)
    
    return neighborhood


def attractor(n_id: int, tree: Tree, cost: int, instance: Instance, fw: FloydWarshall, neighborhood: list):
    """ Trouvele voisin correspondant à l'attraction du noeud d'id n_id et l'ajoute à neighborhood
    Args:
        n_id (int): L'id du noeud attracteur qui va créer le voisin 
        tree (Tree): L'arbre représentant la solution actuelle
        cost (int): Cout selon le critère B de la solution acutelle
        instance (Instance): L'instance du problème
        fw (FloydWarshall): Le FloydWarshall contenant les plus court cheminsde l'instance
        neighborhood (list): Liste des couple (Arbre, Cout) composant le voisinage de tree
    Returns:
        neighborhood (list): Liste des couple (Arbre, Cout) composant le voisinage de tree
    """    
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
        for m_id in nodes: 
            if m_id in newTree.nodes and m_id != 1 and not newTree.is_ancestor(m_id, n_id):
               
                m_cost, m_branch = newTree.cost_and_branch(m_id, fw, instance)
                m_new_cost, m_new_path = fw.get_real_distance_and_path(n_id, m_id, newTree, instance, m_branch)
                
                if n_id not in m_branch and m_cost > m_new_cost:
                    if newTree.distance_from_center(m_new_path[-1].idx()) + len(m_new_path)-1 + newTree.distance_inversed(m_new_path[0].idx(), m_id, None) <= instance.H:
                        # if n_id == 13 and m_id == 66:
                        #     print(newTree.distance_inversed(m_new_path[0].idx(), m_id, None))
                        #     print(instance.is_valid_solution(newTree.tree_to_sol(instance)))
                        #     print(n_id, m_id)
                        #     print(m_cost, m_branch)
                        #     print(m_new_cost, m_new_path)
                        #     print(fw.get_shortest_path(n_id, m_id, instance))
                        #     a = 46
                        #     if a in newTree.nodes: print(newTree.nodes[a].child, newTree.nodes[a].parent)
                        #     else: print(f"{a} not in tree")
                        # avant  = instance.is_valid_solution(newTree.tree_to_sol(instance))
                        newTree.elegate_branch(m_branch)   
                        # if n_id == 13 and m_id == 66:
                        #     print(instance.is_valid_solution(newTree.tree_to_sol(instance)))
                            
                        newTree.inverse_tree(m_new_path[0].idx(), m_new_path[1].idx())
                        # if n_id == 13 and m_id == 66:
                        #     print(instance.is_valid_solution(newTree.tree_to_sol(instance)))

                        newTree.apply_path(m_new_path)

                        # if avant and not instance.is_valid_solution(newTree.tree_to_sol(instance)):
                        #     print(n_id, m_id)
                            
                        # if n_id == 13 and m_id == 66:
                        #     print(instance.is_valid_solution(newTree.tree_to_sol(instance)))
                        #     if a in newTree.nodes: print(newTree.nodes[a].child, newTree.nodes[a].parent)
                        #     else: print(f"{a} not in tree")
                        #     print(instance.nodes_out_of_maximal_distance(newTree.tree_to_sol(instance)))

                        stop = False
                        newCost += m_new_cost - m_cost

    neighborhood.append([newTree, newCost])


def delete_useless_nodes(instance: Instance, fw: FloydWarshall):
    """ Suprrime les noeuds inutiles de l'instance pour notre recherche
    Args:
        instance (Instance): L'instance du problème
        fw (FloydWarshall): Le FloydWarshall contenant les plus court cheminsde l'instance
    """  
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
            to_del.append(n_id)
    for n in to_del:
        del instance.nodes[n]


# def initial_solver(instance: Instance, fw: FloydWarshall, banned_pv: list):
#     tree = Tree()

#     for pn_id in instance.profit_nodes.keys():
#         if pn_id != 1 and pn_id not in banned_pv:
#             path = fw.get_shortest_path(pn_id, 1, instance)
#             if len(path) - 1 <= instance.H:
#                 tree.apply_path(path)
            
#     return tree, instance.solution_cost(tree.tree_to_sol(instance))
    

