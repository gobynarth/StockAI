from pathlib import Path

from live_candidates import load_live_candidates, load_pending_candidates


def test_load_live_candidates_returns_only_fully_configured_live_names():
    repo_root = Path(__file__).resolve().parent.parent

    active = load_live_candidates(repo_root)
    pending = load_pending_candidates(repo_root)

    assert set(active) == {"RIVN", "BITF", "PATH", "HON", "ITW", "NCLH", "NVAX", "WK", "AAL"}
    assert "ENVX" not in active
    assert "TSLA" not in active
    assert active["RIVN"]["trailing_pct"] == 0.03
    assert active["PATH"]["tp"] == 0.20
    assert active["ITW"]["tp"] == 0.20
    assert active["NCLH"]["tp"] == 0.25
    assert active["NVAX"]["tp"] == 0.30
    assert active["WK"]["tp"] == 0.25
    assert active["AAL"]["tp"] == 0.30
    assert {"HD", "ANF", "ISRG", "OXY"} <= {row["ticker"] for row in pending}
