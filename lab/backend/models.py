import pandas as pd
import numpy as np
import dataclasses
import networkx as nx
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from scipy.sparse import coo_matrix
from sklearn.preprocessing import StandardScaler
from spektral.data.graph import Graph
from spektral.data import SingleLoader, Dataset
from spektral.layers import MessagePassing, EdgeConv
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Concatenate, TFSMLayer


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

    def get_graphml(self):
        return nx.generate_graphml(self.graph)

    def get_accounts(self) -> list[Account]:
        accounts = []
        for node, attrs in self.graph.nodes(data=True):
            is_fraud = attrs["is_fraud"]
            customer_id = attrs["customer_id"]
            account = Account(node, customer_id, is_fraud, 0.0)
            accounts.append(account)
        return accounts

    def get_transactions(
        self, filters: dict
    ) -> list[AggregatedTransactionBetweenSenderReceiver]:
        transactions = []
        for sender_id, receiver_id, attrs in self.graph.edges(data=True):
            is_filtered = False
            is_pair_filtered = filters.get("sender_id") and filters.get("receiver_id")

            if is_pair_filtered:
                if (
                    filters["sender_id"] == sender_id
                    and filters["receiver_id"] == receiver_id
                ):
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

    def create_new_transaction(
        self, sender_id: int, receiver_id: int, total_amount: float
    ):
        sender = self.get_account(sender_id)
        receiver = self.get_account(receiver_id)

        if not sender:
            raise ValueError(f"Sender {sender_id} not found")

        if not receiver:
            raise ValueError(f"Receiver {receiver_id} not found")

        pair = (str(sender.id), str(receiver.id))

        if self.graph.has_edge(*pair):
            current_total_amount = self.graph.edges[pair]["total_amount"]
            self.graph.edges[pair]["total_amount"] = current_total_amount + float(
                total_amount
            )
            self.graph.edges[pair]["total_transactions"] += 1
        else:
            raise ValueError(
                f"There is no existing learned transactions between accounts {sender_id} and {receiver_id}."
            )

# Create a custom dataset class
class SingleGraphDataset(Dataset):
    def __init__(self, graph, **kwargs):
        self.graph = graph
        super().__init__(**kwargs)

    def read(self):
        return [self.graph]

class EdgeNodeGCN(Model):
    def __init__(self):
        super().__init__()
        # Masking
        # self.masking = GraphMasking()


        # Node Convolution part using MessagePassing
        self.node_conv = MessagePassing(aggregate='sum')
        self.node_dense = Dense(256, activation='relu')

        # Edge Convolution part
        self.edge_conv = EdgeConv(256, activation='relu')
        self.edge_dense = Dense(128, activation='relu')

        # Combine node and edge features
        self.concat = Concatenate()

        # Final output layer
        self.final_dense1 = Dense(32, activation='relu')
        self.final_dense2 = Dense(1, activation='sigmoid')

    @tf.function
    def call(self, inputs):
        x, a, e = inputs

        # Masking
        # x = self.masking(x)

        # Node convolution
        convoluted_nodes = self.node_conv([x, a, e])  # Apply message passing
        convoluted_nodes = self.node_dense(convoluted_nodes)

        # Edge convolution
        convoluted_edges = self.edge_conv([x, a, e])
        convoluted_edges = self.edge_dense(convoluted_edges)

        # Combine node and edge features
        combined_features = self.concat([convoluted_nodes, convoluted_edges])

        # Final prediction
        out = self.final_dense1(combined_features)
        out = self.final_dense2(out)

        return out

class GNN:

    def __init__(self, graph: nx.Graph):
        print("Loading GNN...")

        # Import TensorFlow model from exported file
        self.graph = graph
        self.model = EdgeNodeGCN()
        sample_input = [
            tf.random.uniform((1, 4)),  # Sample node features
            tf.sparse.SparseTensor(indices=[[0, 0]], values=[1.0], dense_shape=[1, 1]),  # Sample adjacency matrix
            tf.random.uniform((1, 2))  # Sample edge features
        ]
        self.model(sample_input)  # Build the model
        self.model.load_weights('gnn/model_weights.h5')
        # self.model = TFSMLayer("gnn/compiled_model", call_endpoint='serving_default')

        self.edge_labels = np.array(
            [
                0 if attrs["fraud_proportion"] == 0 else 1
                for _, _, attrs in self.graph.edges(data=True)
            ]
        )
        self.labels = np.array(
            [
                0 if attrs["is_fraud"] == 0 else 1
                for _, attrs in self.graph.nodes(data=True)
            ]
        )

        self.a = self._generate_adjency_matrix()
        self.x = self._generate_node_features().numpy()
        self.e = self._generate_edge_features().numpy()

        print("GNN loaded")

    def predict(self, graph):
        self.graph = graph
        self.a = self._generate_adjency_matrix()
        # self.x = self._generate_node_features().numpy()
        self.e = self._generate_edge_features().numpy()
        y = self.labels

        graph = Graph(x=self.x, a=self.a, e=self.e, y=y)

        dataset = SingleGraphDataset(graph)
        loader = SingleLoader(dataset=dataset)

        # Predict
        inputs = loader.load()
        predictions = self.model.predict(inputs, steps=loader.steps_per_epoch)
        return predictions

        

    def _generate_adjency_matrix(self) -> pd.DataFrame:
        print("Generating adjency matrix...")
        # Extract node indices
        node_list = list(self.graph.nodes)
        node_indices = {node: idx for idx, node in enumerate(node_list)}

        # Calculate edge weights
        max_transactions = max(
            [attrs["total_transactions"] for _, _, attrs in self.graph.edges(data=True)]
        )
        min_transactions = min(
            [attrs["total_transactions"] for _, _, attrs in self.graph.edges(data=True)]
        )

        print("Max transactions: ", max_transactions)
        print("Min transactions: ", min_transactions)

        # Max min regulzarization
        for u, v, attrs in self.graph.edges(data=True):
            x = max_transactions - min_transactions
            if x == 0:
                break
            else:
                attrs["weight"] = (attrs["total_transactions"] - min_transactions) / (
                    max_transactions - min_transactions
                )

        # Extract edge indices using the node indices
        edge_indices = np.array(
            [
                (node_indices[u], node_indices[v])
                for u, v, attrs in self.graph.edges(data=True)
            ]
        )
        edge_weights = np.array(
            [attrs["weight"] for _, _, attrs in self.graph.edges(data=True)]
        )

        # Create Adjency matrix
        adjency_matrix_coo_df = pd.DataFrame(
            [], columns=["source", "target", "weight", "label"]
        )
        for ix, edge in enumerate(edge_indices):
            adjency_matrix_coo_df.loc[len(adjency_matrix_coo_df)] = [
                int(edge[0]),
                int(edge[1]),
                edge_weights[ix],
                int(self.edge_labels[ix]),
            ]

        num_nodes = len(self.graph.nodes)

        source_nodes = adjency_matrix_coo_df["source"].apply(lambda x: int(x)).values
        target_nodes = adjency_matrix_coo_df["target"].apply(lambda x: int(x)).values
        weights = adjency_matrix_coo_df["weight"].values

        return coo_matrix(
            (weights, (source_nodes, target_nodes)),
            shape=(num_nodes, num_nodes),
        )

    def _generate_edge_features(self):
        print("Generating edge features...")
        # Sample data
        total_amount_vec = [
            attrs.get("total_amount") for a, b, attrs in self.graph.edges(data=True)
        ]

        # Convert to NumPy array and reshape
        total_amount_vec = np.array(total_amount_vec).reshape(-1, 1)

        # Check for NaN or infinite values
        if np.any(np.isnan(total_amount_vec)) or np.any(np.isinf(total_amount_vec)):
            raise ValueError("Data contains NaN or infinite values")

        # Alternative normalization using StandardScaler
        scaler = StandardScaler()

        scaler.fit(total_amount_vec)
        total_amount_normalized = scaler.transform(total_amount_vec)

        total_transactions_vec = np.array(
            [attrs.get("total_transactions") for a, b, attrs in self.graph.edges(data=True)]
        ).reshape(-1, 1)
        scaler.fit(total_transactions_vec)
        total_transactions_normalized = scaler.transform(total_transactions_vec)
        edge_features = tf.concat(
            [
                total_amount_normalized,
                total_transactions_normalized,
                # fraud_proportion_normalized,
            ],
            axis=-1,
        )
        return edge_features

    def _generate_node_features(self) -> tf.Tensor:
        print("Generating node features...")
        country_vocab = set()
        account_type_vocab = set()
        customer_id_vocab = set()

        for n, attrs in self.graph.nodes(data=True):

            # Preprocess the customer ID
            attrs["customer_id"] = attrs.get("customer_id").replace("_", "")

            country_vocab.add(attrs.get("country", "unknown"))
            account_type_vocab.add(attrs.get("type", "unknown"))
            customer_id_vocab.add(attrs.get("customer_id"))

        print(f"Country vocab size: {len(country_vocab)}")
        print(f"Type vocab size: {len(account_type_vocab)}")
        print(f"Customer ID vocab size: {len(customer_id_vocab)}")


        # Define a TensorFlow encoder
        class NodeAttributeEncoder(tf.keras.layers.Layer):
            def __init__(
                self, country_vocab_size, type_vocab_size, customer_id_vocab_size, output_dim
            ):
                super(NodeAttributeEncoder, self).__init__()
                self.country_embedding = tf.keras.layers.Embedding(
                    country_vocab_size, output_dim
                )
                self.account_type_embedding = tf.keras.layers.Embedding(
                    type_vocab_size, output_dim
                )
                self.customer_id_normalizer = tf.keras.layers.Embedding(
                    customer_id_vocab_size, output_dim
                )

                # Initialize tokenizers
                self.country_tokenizer = Tokenizer(
                    num_words=country_vocab_size, oov_token="<OOV>", split=" "
                )
                self.type_tokenizer = Tokenizer(num_words=type_vocab_size, oov_token="<OOV>", split=" ")
                self.customer_id_tokenizer = Tokenizer(
                    num_words=customer_id_vocab_size, oov_token="<OOV>", split=" "
                )

            def fit_tokenizers(self, country_texts, type_texts, customer_id_texts):
                # Fit tokenizers on the respective input data
                self.country_tokenizer.fit_on_texts(country_texts)
                self.type_tokenizer.fit_on_texts(type_texts)
                self.customer_id_tokenizer.fit_on_texts(customer_id_texts)

            def call(self, inputs):
                country, customer_id, account_type_, is_fraud = inputs

                # Convert text to sequences of integers
                country_seq = self.country_tokenizer.texts_to_sequences([country])
                type_seq = self.type_tokenizer.texts_to_sequences([account_type_])
                customer_id_seq = self.customer_id_tokenizer.texts_to_sequences([customer_id])

                # Convert sequences to tensors
                country_seq = tf.constant(country_seq)
                type_seq = tf.constant(type_seq)
                customer_id_seq = tf.constant(customer_id_seq)

                # Pad sequences to ensure uniform length
                country_seq = tf.keras.preprocessing.sequence.pad_sequences(
                    country_seq, padding="post"
                )
                type_seq = tf.keras.preprocessing.sequence.pad_sequences(
                    type_seq, padding="post"
                )
                customer_id_seq = tf.keras.preprocessing.sequence.pad_sequences(
                    customer_id_seq, padding="post"
                )

                # Embed the sequences
                country_encoded = self.country_embedding(country_seq)
                type_encoded = self.account_type_embedding(type_seq)
                customer_id_encoded = self.customer_id_normalizer(customer_id_seq)


                # Embed is_fraud
                is_fraud_tf= tf.reshape(tf.cast(is_fraud, tf.float32), [1,1,1])
                return tf.concat([country_encoded, customer_id_encoded, type_encoded, is_fraud_tf], axis=-1)


        node_encoder = NodeAttributeEncoder(
            len(country_vocab) + 1,
            len(account_type_vocab) + 1,
            len(customer_id_vocab) + 1,
            output_dim=1,
        )
        node_encoder.fit_tokenizers(
            list(country_vocab), list(account_type_vocab), list(customer_id_vocab)
        )

        # Encode node attributes to feature vectors
        node_features = tf.TensorArray(tf.float32, size=self.graph.number_of_nodes())
        for index, node_obj in enumerate(self.graph.nodes(data=True)):
            node, attrs = node_obj
            encoded_features = node_encoder.call(
                (
                    attrs.get("country", "unknown"),
                    attrs.get("customer_id", "unknown"),
                    attrs.get("type", "unknown"),
                    0 if not attrs.get("is_fraud") else 1
                )
            )
            node_features = node_features.write(index, tf.squeeze(encoded_features, axis=0))

        # Convert TensorArray to Tensor
        node_features_tf = node_features.stack()
        node_features_tf = tf.reshape(node_features_tf, (self.graph.number_of_nodes(), 4))
        return node_features_tf