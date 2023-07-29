import warnings

import pytest


@pytest.fixture(autouse=True)
def suppress_pgmpy_warnings():
    warnings.filterwarnings("ignore", "DeprecationWarning:pgmpy.*")
    warnings.filterwarnings("ignore", "DeprecationWarning:numpy.*")
    warnings.filterwarnings("ignore", "DeprecationWarning:pkg_resources.*")
    warnings.filterwarnings(
        "ignore", category=UserWarning, message="Probability values don't exactly sum to 1."
    )
