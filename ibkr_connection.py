from dataclasses import dataclass


@dataclass
class IbkrConnectionResult:
    ib: object | None
    read_only: bool
    accounts: list[str]


def connect_ibkr_session(
    ib_factory,
    host,
    port,
    client_id,
    timeout=10,
    force_read_only=None,
    logger=print,
):
    """Connect to IBKR, retrying in read-only mode when order sync hangs."""
    if force_read_only is True:
        attempts = [True]
    elif force_read_only is False:
        attempts = [False]
    else:
        attempts = [False, True]

    last_error = None
    for idx, read_only in enumerate(attempts):
        ib = ib_factory()
        try:
            ib.connect(host, port, clientId=client_id, timeout=timeout, readonly=read_only)
            accounts = list(ib.managedAccounts())
            mode = "read-only" if read_only else "trading"
            logger(f"  IBKR connected on {host}:{port} ({mode}; accounts: {accounts})")
            return IbkrConnectionResult(ib=ib, read_only=read_only, accounts=accounts)
        except Exception as exc:
            last_error = exc
            try:
                ib.disconnect()
            except Exception:
                pass
            if idx == 0 and len(attempts) > 1:
                logger(f"  IBKR standard sync failed ({exc}) -- retrying in read-only mode")

    logger(f"  IBKR not available ({last_error}) -- orders will NOT be submitted")
    return IbkrConnectionResult(ib=None, read_only=False, accounts=[])
