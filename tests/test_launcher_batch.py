from pathlib import Path


def test_run_daily_signal_starts_ibc_without_blocking():
    launcher = Path("run_daily_signal.bat").read_text()

    assert 'start "" /b cmd /c ""%IBC_STARTER%" /INLINE /NOICON"' in launcher
    assert 'call "%IBC_STARTER%" /INLINE /NOICON' not in launcher
