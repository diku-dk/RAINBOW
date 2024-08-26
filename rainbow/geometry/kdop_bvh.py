import rainbow.geometry.kdop as KDOP
import math
from collections import deque


class Node:
    """
    A node class indicating a node in a BVH tree.
    """

    UNDEFINED = -1

    def __init__(self):
        self.volume = None
        self.parent = Node.UNDEFINED
        self.start = Node.UNDEFINED
        self.end = Node.UNDEFINED


class SubTree:
    def __init__(self, size):
        self.nodes = [Node() for _ in range(size)]
        self.height = 0


class Tree:
    def __init__(self):
        self.root = None
        self.chunks = []


def _is_root(node):
    return (
        node.parent == Node.UNDEFINED
        and node.start != Node.UNDEFINED
        and node.end != Node.UNDEFINED
    )


def _is_leaf(node):
    return node.start != Node.UNDEFINED and node.start == node.end


def _is_undefined(node):
    return (
        node.parent == Node.UNDEFINED
        and node.start == Node.UNDEFINED
        and node.end == Node.UNDEFINED
    )


def _refit_sub_tree(V, T, sub_tree, K, envelope=0.0) -> None:
    """
    This function refits a subtree by iterating through every node in a bottom up
    approach. Each node will then be updated based on its children.

    :param V:             The current coordinates of the object
    :param T:             The elements (triangles/tetrahedrons) of the object
    :param sub_tree:      The BVH subtree
    :param K:             The number of intervals for the KDOPs
    :param envelope:      The size of the envelope used to enlarge the volume. If a tight fit is wanted
                          one should use the value zero.
    """
    for node in reversed(sub_tree.nodes):
        if _is_undefined(node):
            continue
        if _is_leaf(node):
            vertices = V[T[node.start]]
            node.volume = KDOP.make(K, vertices, envelope)
        else:
            left_child = sub_tree.nodes[node.start]
            right_child = sub_tree.nodes[node.end]
            node.volume = KDOP.union(left_child.volume, right_child.volume)


def _make_sub_tree(parent_idx, free_idx, first, last, sub_tree):
    """
    Creates a bvh subtree recursively.
    This function creates the BVH tree in a top-down in order traversal scheme. It builds the tree one node at the
    time. When a node is created the function keeps track of the range of data-elements that are covered by this
    node. In recursive calls the range of data is split into two halves until only one element exist in the range. When
    this happens a leaf-node has been reached.

    A node has a left and right index member that points to its left and right child nodes. Unless a leaf nodes is
    encountered. In this case the left and right indices holds the value of the single data element that is covered
    by the leaf node.

    If for instance the data input is a tetrahedral mesh the the data-elements covered by the BVH would be
    tetrahedrons. For a leaf node the left and right indices would both hold the index value of the tetrahedron that
    is bounded by that leaf node.

    :param parent_idx:  The index of the parent node to the given subtree.
    :param free_idx:    The index of the next free node.
    :param first:       The first index of the data elements that is covered by this node.
    :param last:        The last index of the data elements that is covered by this node.
    :param sub_tree:    The subtree structure to fill out.
    :return:            The height of the subtree and the index of the next free node in the subtree.
    """
    N = last - first + 1
    if N > 1:
        left_idx = free_idx
        right_idx = free_idx + 1
        free_idx += 2

        sub_tree.nodes[parent_idx].start = left_idx
        sub_tree.nodes[parent_idx].end = right_idx

        sub_tree.nodes[left_idx].parent = parent_idx
        sub_tree.nodes[right_idx].parent = parent_idx

        mid_idx = math.floor((first + last) / 2.0)

        left_height, free_idx = _make_sub_tree(
            left_idx, free_idx, first, mid_idx, sub_tree
        )
        right_height, free_idx = _make_sub_tree(
            right_idx, free_idx, mid_idx + 1, last, sub_tree
        )

        return max(left_height, right_height) + 1, free_idx
    else:
        sub_tree.nodes[parent_idx].start = first
        sub_tree.nodes[parent_idx].end = first
        return 0, free_idx


def _iterative_sub_tree_traversal(sub_tree_A, sub_tree_B, results) -> None:
    """
    This function performs an iterative tandem traversal of subtrees as described
    in "Physics-based animation" by Erleben, Kenny, Jon Sporring, et al.

    :param sub_tree_A:    The first BVH subtree
    :param sub_tree_B:    The second BVH subtree
    :param results:       A list of primitive index tuples corresponding to all
                         the overlapping leaf volumes that has been detected.
    """
    node_A = sub_tree_A.nodes[0]
    node_B = sub_tree_B.nodes[0]

    Q = deque()
    Q.append(node_A)
    Q.append(node_B)

    while Q:
        node_A = Q.popleft()
        node_B = Q.popleft()

        if not KDOP.overlap(node_A.volume, node_B.volume):
            continue

        A_is_leaf = _is_leaf(node_A)
        B_is_leaf = _is_leaf(node_B)

        if A_is_leaf and B_is_leaf:
            results.append((node_A.start, node_B.start))
            continue

        if not A_is_leaf and not B_is_leaf:
            for i in range(node_A.start, node_A.end + 1):
                child_A = sub_tree_A.nodes[i]
                for j in range(node_B.start, node_B.end + 1):
                    child_B = sub_tree_B.nodes[j]
                    Q.append(child_A)
                    Q.append(child_B)
            continue

        if not A_is_leaf:
            for i in range(node_A.start, node_A.end + 1):
                child_A = sub_tree_A.nodes[i]
                Q.append(child_A)
                Q.append(node_B)
            continue

        if not B_is_leaf:
            for i in range(node_B.start, node_B.end + 1):
                child_B = sub_tree_B.nodes[i]
                Q.append(node_A)
                Q.append(child_B)


def make_bvh(X, T, K, N, envelope=0.0):
    """
    This function creates a bvh tree bounding an object

    :param X:             The spatial nodes of the object
    :param T:             The elements (triangles/tetrahedrons) of the object
    :param K:             The number of intervals to be created for the KDOPs
    :param N:             The size of each subtree
    :param envelope:      The size of the envelope used to enlarge the volume. If a tight fit is wanted
                          one should use the value zero.
    :return:              A KDOP BVH datastructure
    """
    tree = Tree()
    L = int((N + 1) / 2)  # Number of leaves in a balanced binary tree with N nodes
    M = len(T)  # Number of data elements in the mesh
    C = math.ceil(M / L)  # Number of chunks to divide the mesh into

    for c in range(C):
        sub_tree = SubTree(N)
        # Compute the range of data elements in the mesh that is covered by this subtree
        first = c * L
        last = min(first + L - 1, M - 1)
        # Next generate the subtree
        root_idx = 0
        free_idx = 1
        sub_tree.height = _make_sub_tree(root_idx, free_idx, first, last, sub_tree)
        # Finally store the sub-tree
        tree.chunks.append(sub_tree)

    refit_bvh(X, T, tree, K, envelope)

    return tree


def refit_bvh(V, T, tree, K, envelope=0.0) -> None:
    """
    This function refits a BVH tree such that all nodes are updated to the current
    coordinates of the object it encompasses.

    :param V:                 The current coordinates of the object
    :param T:                 The elements of the object
    :param tree:              The BVH tree
    :param K:                 The number of intervals for the KDOPs
    :param envelope:          The size of the envelope used to enlarge the volume. If a tight fit is wanted
                              one should use the value zero.
    """
    for subtree in tree.chunks:
        _refit_sub_tree(V, T, subtree, K, envelope)
    tree.root = KDOP.KDOP(K)
    for subtree in tree.chunks:
        tree.root = KDOP.union(tree.root, subtree.nodes[0].volume)


def traversal(tree_A, tree_B):
    """
    This function iterates through each combination of subtree pairs between two
    BVH trees to test for overlaps.

    :param tree_A:        The first BVH tree
    :param tree_B:        The second BVH tree
    :return:              An instance of the Overlaps class containing information about
                          overlaps between the given BVH trees. The results of the
                          traversal will be stored here.
    """
    results = []
    if not KDOP.overlap(tree_A.root, tree_B.root):
        return results
    for sub_tree_A in tree_A.chunks:
        for sub_tree_B in tree_B.chunks:
            _iterative_sub_tree_traversal(sub_tree_A, sub_tree_B, results)
    return results
