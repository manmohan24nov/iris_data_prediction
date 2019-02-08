import pandas as pd

def main():
    raw_data = pd.read_csv('bezdekIris.data',
                           sep=',',
                           names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width','class'])
    return raw_data

if __name__ == '__main__':
    main()
