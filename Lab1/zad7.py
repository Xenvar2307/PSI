word_input = input("Podaj słowo:")


def space_out_word(word):
    result = ""
    if len(word) > 0:
        result += word[0]

    for letter in word[1:]:
        result += " " + letter

    return result


print(space_out_word(word_input))
