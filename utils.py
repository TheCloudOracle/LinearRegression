def train_test_split(X, y, train_size):
    train_size *= len(X)
    train_size = int(train_size)
    return X[:train_size], X[train_size:], y[:train_size], y[train_size:]

