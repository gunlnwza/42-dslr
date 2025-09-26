import numpy as np

def ft_count(values: np.ndarray) -> float:
    return np.sum(~np.isnan(values))


def ft_mean(values: np.ndarray) -> float:

    clean_value = values[~np.isnan(values)]

    if clean_value.size == 0:
        return np.nan
    
    return (np.sum(clean_value) / clean_value.size)


def ft_stddev(values: np.ndarray) -> float:

    clean_value = values[~np.isnan(values)]
    size = clean_value.size

    if size < 2:
        return np.nan

    mean = ft_mean(clean_value)
    squared_difference = (clean_value - mean) ** 2
    varience = np.sum(squared_difference) / (size - 1)

    return np.sqrt(varience)


def ft_percentile(values: np.ndarray, n_th_percentile: float) -> float:

    clean_value = values[~np.isnan(values)]
    size = clean_value.size

    if size == 0:
        return np.nan
    
    sorted_values = np.sort(clean_value)
    idx_float = (size - 1) * (n_th_percentile / 100.0)

    if idx_float < 0.0:
        return sorted_values[0]

    if idx_float >= size - 1:
        return sorted_values[-1]
    
    idx = int(idx_float)
    f = idx_float - idx

    return (sorted_values[idx] + (f * (sorted_values[idx + 1] - sorted_values[idx])))


def ft_min(values: np.ndarray) -> float:

    clean_value = values[~np.isnan(values)]
    size = clean_value.size

    if size == 0:
        return np.nan

    minimum = clean_value[0]

    for x in clean_value[1:]:
        if x < minimum:
            minimum = x

    return minimum


def ft_max(values: np.ndarray) -> float:

    clean_value = values[~np.isnan(values)]
    size = clean_value.size

    if size == 0:
        return np.nan

    maximum = clean_value[0]

    for x in clean_value[1:]:
        if x > maximum:
            maximum = x

    return maximum


def ft_variance(values: np.ndarray) -> float:

    """Calculates sample variance (stddev squared)."""

    clean_value = values[~np.isnan(values)]
    size = clean_value.size

    if size < 2:
        return np.nan
    
    mean = ft_mean(clean_value)
    squared_difference = (clean_value - mean) ** 2
    return np.sum(squared_difference) / (clean_value.size - 1)


def ft_iqr(values: np.ndarray) -> float:

    """Calculates Interquartile Range (75th percentile - 25th percentile)."""

    clean_value = values[~np.isnan(values)]
    size = clean_value.size

    if size == 0:
        return np.nan
    
    q3 = ft_percentile(clean_value, 75)
    q1 = ft_percentile(clean_value, 25)
    
    return q3 - q1


def ft_moment(values: np.ndarray, k: int) -> float:

    """
    Calculates the k-th central moment (unadjusted population estimate).
    m_k = (1/n) * sum((x_i - mean)^k)
    """

    clean_value = values[~np.isnan(values)]
    size = clean_value.size
    
    if size < 2:
        return np.nan

    mean = ft_mean(clean_value)
    k_th_difference = (clean_value - mean) ** k

    return np.sum(k_th_difference) / clean_value.size


def ft_skewness(values: np.ndarray) -> float:

    """
    Calculates the unadjusted sample skewness (3rd standardized moment).
    Skew = m_3 / m_2^(3/2)
    """

    clean_value = values[~np.isnan(values)]
    size = clean_value.size

    if size < 3:
        return np.nan
    
    m3 = ft_moment(clean_value, 3)
    m2 = ft_moment(clean_value, 2)
    
    if m2 == 0:
        return np.nan 

    return m3 / (m2 ** 1.5)


def ft_kurtosis(values: np.ndarray) -> float:

    """
    Calculates the unadjusted excess kurtosis (4th standardized moment - 3).
    Kurtosis = (m_4 / m_2^2) - 3
    """

    clean_value = values[~np.isnan(values)]
    size = clean_value.size

    if size < 4:
        return np.nan
    
    m4 = ft_moment(clean_value, 4)
    m2 = ft_moment(clean_value, 2)

    # Check for zero variance
    if m2 == 0:
        return np.nan 

    # 4th standardized moment minus 3 (excess kurtosis)
    return (m4 / (m2 ** 2)) - 3
