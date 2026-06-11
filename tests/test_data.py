import pandas as pd
import pytest

from house_price.data import validate_columns


def test_validate_columns_reports_missing_columns() -> None:
    data = pd.DataFrame({"Id": [1]})

    with pytest.raises(ValueError, match="SalePrice"):
        validate_columns(data, {"Id", "SalePrice"})

