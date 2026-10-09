"""In-process Flower Grid: runs the real ServerApp logic and ClientApp ``train`` handler without Ray.

Used by the tests and as the ``inprocess`` backend. Clients are called sequentially in a fixed order, which makes a
run bit-for-bit repeatable; it is not a model of network behaviour (no failures, no stragglers).
"""

from __future__ import annotations

from collections.abc import Iterable

from flwr.app import Context, Message, RecordDict

from experiments.stacked_head import client as client_module


class LocalGrid:
    """Minimal ``Grid`` surface needed by the strategy: node ids and ``send_and_receive``."""

    def __init__(self, num_clients: int, run_config: dict, client_app=client_module) -> None:
        self.num_clients = num_clients
        self.run_config = run_config
        self.client_app = client_app

    def get_node_ids(self) -> list[int]:
        return list(range(1, self.num_clients + 1))

    def send_and_receive(self, messages: Iterable[Message], timeout: float | None = None) -> list[Message]:
        replies = []
        for message in messages:
            if message.metadata.message_type != "train":
                continue
            node_id = int(message.metadata.dst_node_id)
            context = Context(run_id=1, node_id=node_id, node_config={"partition-id": node_id - 1}, state=RecordDict(), run_config=self.run_config)
            replies.append(self.client_app.train(message, context))
        return replies
