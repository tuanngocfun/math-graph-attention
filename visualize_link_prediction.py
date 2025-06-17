import networkx as nx
import matplotlib.pyplot as plt


def build_graph():
    """Create a sample graph for demonstration."""
    # simple undirected graph
    G = nx.Graph()
    edges = [(1, 2), (1, 3), (2, 3), (3, 4), (4, 5)]
    G.add_edges_from(edges)
    return G


def predict_links(G, top_k=2):
    """Predict links using common neighbors heuristic."""
    scores = []
    for u, v in nx.non_edges(G):
        cn = len(list(nx.common_neighbors(G, u, v)))
        if cn > 0:
            scores.append(((u, v), cn))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [edge for edge, _ in scores[:top_k]]


def visualize(G, predicted_edges):
    """Plot original graph and graph with predicted edges."""
    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(10, 5))

    # original graph
    plt.subplot(1, 2, 1)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray')
    plt.title('Original Graph')

    # graph with predicted edges
    G_pred = G.copy()
    G_pred.add_edges_from(predicted_edges)

    plt.subplot(1, 2, 2)
    nx.draw(G_pred, pos, with_labels=True, node_color='lightblue', edge_color='gray')
    nx.draw_networkx_edges(G_pred, pos, edgelist=predicted_edges, edge_color='red', style='dashed')
    plt.title('With Predicted Links')

    plt.tight_layout()
    plt.show()


def main():
    G = build_graph()
    predicted = predict_links(G)
    visualize(G, predicted)


if __name__ == "__main__":
    main()
