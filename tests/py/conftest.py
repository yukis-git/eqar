import pytest

import erp_synth as S

W = 1920


@pytest.fixture(scope="session")
def frame():
    """1920x960 gray frame, marker id 7 at (lon -100, lat 20, half 8, roll 15)."""
    return S.make_frame(W, [dict(id=7, lon=-100, lat=20, half=8, roll=15)], noise=2, blur=0.6)


GT = dict(id=7, lon=-100, lat=20, half=8, roll=15)
