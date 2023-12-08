import unittest
import os
import sys
import numpy as np
import networkx as nx

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rainbow.simulators.proximal_contact.blocking as BLOCKING
import rainbow.util.test_tools as TEST


class TestBlocking(unittest.TestCase):

    def setUp(self):
        self.K = 5
        self.G = nx.Graph()
        self.G.add_nodes_from(np.arange(self.K))

    def test_random_graph_coloring(self):
        # Create a graph: 0-1-2-3-4, here the number is the node id.
        self.G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
        color_groups = BLOCKING.random_graph_coloring(self.G)

        # Check if the color groups are valid
        # The color groups should be: {0,2,4}, {1, 3}
        expect_color_groups = {0: [0, 2, 4], 1: [1, 3]}
        for c in color_groups.values():
            if len(c) == 3:
                TEST.is_array_equal(c, expect_color_groups[0])
            else:
                TEST.is_array_equal(c, expect_color_groups[1])


if __name__ == "__main__":
    unittest.main()