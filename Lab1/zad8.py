word_input = input("Podaj słowo:")


def reverse_word(word):
    result = ""

    for letter in word:
        result = letter + result

    return result

# could be done by [::-1]

print(reverse_word(word_input))
