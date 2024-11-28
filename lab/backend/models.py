import dataclasses
import networkx as nx


@dataclasses.dataclass
class Account:
    id: int
    customer_id: str
    fraudulent: bool
    fraud_probability: float

@dataclasses.dataclass
class AggregatedTransactionBetweenSenderReceiver:
    sender: Account
    receiver: Account
    total_amount: float
    total_transactions: int
    fraud_probability: float


class GraphDatabase:

    def __init__(self):
        print("Loading graph...")
        with open("gnn/data/graph.graphml", "r") as f:
            self.graph = nx.read_graphml(f)
            print("Graph loaded")
            print(f"Nodes: {self.graph.number_of_nodes()}")
            print(f"Edges: {self.graph.number_of_edges()}")

    def get_accounts(self) -> list[Account]:
        accounts = []
        for node, attrs in self.graph.nodes(data=True):
            is_fraud = attrs["is_fraud"]
            customer_id = attrs["customer_id"]
            account = Account(node, customer_id,  is_fraud, 0.0)
            accounts.append(account)
        return accounts

    def get_transactions(self, filters: dict) -> list[AggregatedTransactionBetweenSenderReceiver]:
        transactions = []
        for sender_id, receiver_id, attrs in self.graph.edges(data=True):
            is_filtered = False
            is_pair_filtered = filters.get("sender_id") and filters.get("receiver_id")

            if is_pair_filtered:
                if filters["sender_id"] == sender_id and filters["receiver_id"] == receiver_id:
                    is_filtered = True
            else:
                if filters.get("sender_id") and filters["sender_id"] == sender_id:
                    is_filtered = True

                if filters.get("receiver_id") and filters["receiver_id"] != receiver_id:
                    is_filtered = True

            if not is_filtered:
                continue

            sender = self.get_account(sender_id)
            receiver = self.get_account(receiver_id)

            sender_account = Account(sender_id, sender.customer_id, False, 0.0)
            receiver_account = Account(receiver_id, receiver.customer_id, False, 0.0)
            total_amount = attrs["total_amount"]
            total_transactions = attrs["total_transactions"]
            fraud_probability = 0.0
            transaction = AggregatedTransactionBetweenSenderReceiver(
                sender_account,
                receiver_account,
                total_amount,
                total_transactions,
                fraud_probability,
            )
            transactions.append(transaction)
        return transactions

    def get_account(self, id: int) -> Account:
        if self.graph.has_node(str(id)):
            node = self.graph.nodes[str(id)]
            is_fraud = node["is_fraud"]
            customer_id = node["customer_id"]
            return Account(node["id"], customer_id, is_fraud, 0.0)

    def create_new_transaction(self, sender_id: int, receiver_id: int, total_amount: float):
        sender = self.get_account(sender_id)
        receiver = self.get_account(receiver_id)

        if not sender:
            raise ValueError(f"Sender {sender_id} not found")
        
        if not receiver:
            raise ValueError(f"Receiver {receiver_id} not found")

        pair = (str(sender.id), str(receiver.id))
        self.graph.edges[pair]["total_amount"] += total_amount
        self.graph.edges[pair]["total_transactions"] += 1
