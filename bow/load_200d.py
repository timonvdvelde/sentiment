def calc_word_offset(file):
    dictionary = {}
    file = open(file, "r")
    for line in file:
        word = line.split()[0]
        dictionary[word]

    return dictionary
