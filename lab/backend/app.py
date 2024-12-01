from flask import Flask, jsonify, request
from models import GraphDatabase, GNN

app = Flask(__name__)


@app.route("/")
def home():
    return open("lab/frontend/index.html").read()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)

graph_database = GraphDatabase()
gnn_model = GNN(graph_database.graph)


@app.route("/accounts")
def get_accounts():
    accounts = graph_database.get_accounts()
    return jsonify(accounts)


@app.route("/transactions")
def get_transactions(sender_id=None, receiver_id=None):
    sender_id = request.args.get("sender_id")
    receiver_id = request.args.get("receiver_id")

    if sender_id is None and receiver_id is None:
        return "Please provide sender_id or receiver_id as query parameter", 400

    transactions = graph_database.get_transactions(
        filters={"sender_id": sender_id, "receiver_id": receiver_id}
    )
    return jsonify(transactions)


@app.route("/transactions/new", methods=["POST"])
def get_transactions_by_sender_receiver():
    payload = request.get_json()
    sender_id = payload["sender_id"]
    receiver_id = payload["receiver_id"]
    total_amount = payload["total_amount"]

    # Update graph with new transaction
    try:
        graph = graph_database.graph
        graph_database.create_new_transaction(sender_id, receiver_id, total_amount)

        predicitons = gnn_model.predict(graph)

        # Find Pair prediction
        node_list = list(graph.nodes)
        node_indices = {node: idx for idx, node in enumerate(node_list)}

        node_a_ix = node_indices[str(sender_id)]
        node_b_ix = node_indices[str(receiver_id)]

        a_fraud = float(predicitons.flatten()[node_a_ix])
        b_fraud = float(predicitons.flatten()[node_b_ix])

        return jsonify(
            {
                "nodes": {"sender": a_fraud, "receiver": b_fraud},
                "is_fraud": bool(a_fraud > 0.5 or b_fraud > 0.5),
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/graphml")
def get_graphml():
    # Return the graph in GraphML format
    return graph_database.get_graphml()
