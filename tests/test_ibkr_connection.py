from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ibkr_connection import connect_ibkr_session


class FakeIB:
    def __init__(self, fail_readwrite=False):
        self.fail_readwrite = fail_readwrite
        self.connect_calls = []
        self.disconnected = False

    def connect(self, host, port, clientId, timeout, readonly=False):
        self.connect_calls.append(
            {
                "host": host,
                "port": port,
                "clientId": clientId,
                "timeout": timeout,
                "readonly": readonly,
            }
        )
        if self.fail_readwrite and not readonly:
            raise TimeoutError("completed orders request timed out")

    def managedAccounts(self):
        return ["DUQ211124"]

    def disconnect(self):
        self.disconnected = True


def test_connect_ibkr_session_retries_in_read_only_mode_after_rw_timeout():
    created = []

    def factory():
        ib = FakeIB(fail_readwrite=True)
        created.append(ib)
        return ib

    messages = []
    result = connect_ibkr_session(
        ib_factory=factory,
        host="127.0.0.1",
        port=4002,
        client_id=1,
        timeout=10,
        logger=messages.append,
    )

    assert result.ib is created[1]
    assert result.read_only is True
    assert created[0].connect_calls == [
        {"host": "127.0.0.1", "port": 4002, "clientId": 1, "timeout": 10, "readonly": False}
    ]
    assert created[0].disconnected is True
    assert created[1].connect_calls == [
        {"host": "127.0.0.1", "port": 4002, "clientId": 1, "timeout": 10, "readonly": True}
    ]
    assert any("retrying in read-only mode" in message for message in messages)


def test_connect_ibkr_session_honors_forced_read_only_mode():
    created = []

    def factory():
        ib = FakeIB(fail_readwrite=False)
        created.append(ib)
        return ib

    result = connect_ibkr_session(
        ib_factory=factory,
        host="127.0.0.1",
        port=4002,
        client_id=7,
        timeout=5,
        force_read_only=True,
        logger=lambda _: None,
    )

    assert result.read_only is True
    assert created[0].connect_calls == [
        {"host": "127.0.0.1", "port": 4002, "clientId": 7, "timeout": 5, "readonly": True}
    ]
