import numpy as np
from io import open

def make_weights(embed_size, glove_path, lang):
    print("Making weights...")
    vocab_size = len(lang.tok_to_idx)
    sd = 1/np.sqrt(embed_size)
    weights = np.random.normal(0, scale=sd, size=[vocab_size, embed_size])
    weights = weights.astype(np.float32)

    debug = False
    if(debug):
        print("Debug weights done!")
        return weights
    with open(glove_path, encoding="utf-8", mode="r") as textFile:
        for line in textFile:
            line = line.split()
            word = line[0]

            id = lang.tok_to_idx.get(word, None)
            if id is not None:
                try:
                    weights[id] = np.array(line[1:], dtype=np.float32)
                except ValueError:
                    continue
    print("Done!")
    return weights

