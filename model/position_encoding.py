# -*-coding:utf-8-*-
#知识嵌入的位置编码
import numpy as np
def knowledge_embedded_PE():
    num_node = 50
    self_link = [(i, i) for i in range(num_node)]
    neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                                  (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                                  (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                                  (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                                  (22, 23), (23, 8), (24, 25), (25, 12)]
    Krelation_base = [(24, 29), (22, 49), (24, 47), (22, 47), (2, 27),
                                 (24, 49), (49, 4), (24, 27), (22, 27), (49, 2),
                                 (47, 2), (24, 34), (24, 30), (22, 34), (22, 30),
                                 (49, 9), (49, 5), (47, 9), (47, 5), (20, 27),
                                 (16, 27), (45, 2), (41, 2)]
    relation_link = [(i - 1, j - 1) for (i, j) in Krelation_base]
    neighbor_2base = [(i + 25, j + 25) for (i, j) in neighbor_1base]
    neighbor_1link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
    neighbor_2link = [(i - 1, j - 1) for (i, j) in neighbor_2base]

    edge = self_link + neighbor_1link + neighbor_2link+relation_link
    center = 2 - 1
    A = np.zeros((num_node, num_node))
    A1 = np.zeros((num_node+10, num_node+10))
    for i, j in edge:
            A[j, i] = 1
            A[i, j] = 1
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    A1[:50,:50] = AD

    index = [2, 3, 8, 20, 4, 24, 23, 11, 10, 9, 22, 21, 7, 6, 5, 19, 18, 17, 16, 53, 0, 1, 50, 51, 52, 54, 15, 14,
             13, 12,27,28,33,45,29,49,48,36,35,34,47, 46, 32, 31, 30, 44, 43, 42, 41, 58, 25, 26, 55, 56,57, 59, 40, 39, 38, 37]
    A1 = A1[index, :]
    A1 = A1[:, index]
    return A1
