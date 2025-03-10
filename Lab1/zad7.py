word_input = input("Podaj słowo:")


def space_out_word(word):
    result = ""

    for letter in word[0:]:
        result += " " + letter

    result += " "
    return result


print(space_out_word(word_input))
