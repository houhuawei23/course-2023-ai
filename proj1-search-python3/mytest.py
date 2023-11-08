from search import Node, traceBackNode

if __name__ == '__main__':
    node1 = Node(1, None, None, 0)
    node2 = Node(2, node1, 'r', 1)
    node3 = Node(3, node2, 'r', 2)
    print(traceBackNode(node3))