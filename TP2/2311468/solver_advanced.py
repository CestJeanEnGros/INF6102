from utils import Node, Edge, Instance, Solution, FloydWarshall, TreeNode, Tree
import numpy as np
import pickle
import random as rd
import time 
import copy
from tqdm import tqdm

START_TIME = None
END_TIME = None

def solve(instance: Instance) -> Solution:
    """ Trouve une solution optimisée du problème
    Args:
        instance (Instance): L'instance du problème
    Returns:
        (Solution): Un objet Solution de l'instance
    """
    # Setup
    global START_TIME 
    global END_TIME
    START_TIME = time.time()
    if instance.M < 2000: END_TIME = 280
    else: END_TIME = 580

    # if instance.B == 85:
    #     with open("instanceA.pkl", "rb") as fichier:
    #         fw = pickle.load(fichier)
    # elif instance.B == 103:
    #     with open("instanceB.pkl", "rb") as fichier:
    #         fw = pickle.load(fichier)
    # elif instance.B == 342:
    #     with open("instanceC.pkl", "rb") as fichier:
    #         fw = pickle.load(fichier)
    # elif instance.B == 272:
    #     with open("instanceD.pkl", "rb") as fichier:
    #         fw = pickle.load(fichier)
    # elif instance.B == 136:
    #     with open("instanceE.pkl", "rb") as fichier:
    #         fw = pickle.load(fichier)
    # elif instance.B == 345:
    #     with open("instanceF.pkl", "rb") as fichier:
    #         fw = pickle.load(fichier)

    tt = time.time()
    fw = FloydWarshall(instance)
    fw.floyd_warshall(instance)
    print(f"Temps d'exec de FW : {time.time() -  tt}")

    # Supression des noeuds inutiles
    delete_useless_nodes(instance, fw)

    # Initialisation de la solution constructive
    tree = Tree()
    cost = 0

    # Ajout successif des PV à la solution
    stop = False
    while not stop:
        if time.time() - START_TIME > END_TIME:
            print("NO MORE TIME")
            break
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
                    newTree = copy.deepcopy(tree)
                    newTree.apply_path(best_path)
                    newCost = cost + best_cost

                    newTree, newCost = local_search(newTree, newCost, instance, fw)
 
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
        if time.time() - START_TIME > END_TIME:
            print("NO MORE TIME")
            break
        stop = True
        neighborhood = find_neighborhood(tree, cost, instance, fw)
        neighborhood.sort(key=lambda x: x[1])

        newTree, newCost = neighborhood[0][0], neighborhood[0][1]
        if newCost < cost:
            stop = False
            tree = copy.deepcopy(newTree)
            cost = newCost
            print(f"New record {newCost}")

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
        if time.time() - START_TIME > END_TIME:
            print("NO MORE TIME")
            break
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
        if time.time() - START_TIME > END_TIME:
            print("NO MORE TIME")
            break
        stop = True
        nodes = list(instance.nodes.keys())
        for m_id in nodes: 
            if time.time() - START_TIME > END_TIME:
                print("NO MORE TIME")
                break
            if m_id in newTree.nodes and m_id != 1 and not newTree.is_ancestor(m_id, n_id):
               
                m_cost, m_branch = newTree.cost_and_branch(m_id, fw, instance)
                m_new_cost, m_new_path = fw.get_real_distance_and_path(n_id, m_id, newTree, instance, m_branch)
                
                if n_id not in m_branch and m_cost > m_new_cost:
                    if newTree.distance_from_center(m_new_path[-1].idx()) + len(m_new_path)-1 + newTree.distance_inversed(m_new_path[0].idx(), m_id, None) <= instance.H:

                        newTree.elegate_branch(m_branch)   
                        newTree.inverse_tree(m_new_path[0].idx(), m_new_path[1].idx())
                        newTree.apply_path(m_new_path)

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

