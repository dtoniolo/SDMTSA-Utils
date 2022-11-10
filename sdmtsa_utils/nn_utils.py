from tensorflow.data import Dataset


def gen_to_tensor(generator):
    dataset = Dataset.from_generator(generator)
    lists = list()
    for i, batch in enumerate(dataset)
