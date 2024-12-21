import numpy as np

def one_hot_encode(text, char_to_int):
    return [char_to_int[char] for char in text]

# Example: Map each character to an integer
chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
char_to_int = {char: i for i, char in enumerate(chars)}

# One-hot encode the text
text = "ABCD1"
encoded_text = one_hot_encode(text, char_to_int)
print(encoded_text)  # [0, 1, 2, 3, 34]
