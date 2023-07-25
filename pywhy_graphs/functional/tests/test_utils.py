import networkx as nx

from pywhy_graphs.functional.utils import set_node_attributes_with_G


# Sample test graph with known node attributes
def create_sample_graph():
    G = nx.Graph()
    G.add_node(1, attr1="value1", attr2="value2")
    G.add_node(2, attr1="value3", attr2="value4")
    G.add_edge(1, 2)
    return G


def test_set_node_attributes_with_G():
    # Create the source and target graphs
    G_source = create_sample_graph()
    G_target = create_sample_graph()

    # Set some attributes to the source node that are not present in the target node
    G_source.nodes[1]["attr3"] = "new_value"

    # Call the function to update target node attributes using the source graph
    updated_G = set_node_attributes_with_G(G_target, G_source, 1)

    # Verify if the target node attributes have been updated correctly
    assert updated_G.nodes[1]["attr1"] == "value1"
    assert updated_G.nodes[1]["attr2"] == "value2"
    assert updated_G.nodes[1]["attr3"] == "new_value"

    # Make sure other node attributes in the target graph are unchanged
    assert updated_G.nodes[2]["attr1"] == "value3"
    assert updated_G.nodes[2]["attr2"] == "value4"
