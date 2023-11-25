
import numpy as np
import pytest
from BCI import calculate_extreme_statistics_iqr

@pytest.fixture
def sample_data():
    return np.array([3, 75, 122, 167, 228, 400, 1100])

def test_calculate_extreme_statistics_iqr_with_default_multiplier(sample_data):
    result = calculate_extreme_statistics_iqr(sample_data, multiplier=1.5)
    assert result['Number of Extreme Values'] == 1
    
    

def test_calculate_extreme_statistics_iqr_with_custom_multiplier(sample_data):
    result = calculate_extreme_statistics_iqr(sample_data, multiplier=2.0)
    assert result['Number of Extreme Values'] == 0
    