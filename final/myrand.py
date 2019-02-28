import random
with open("submission.csv", "wb") as fp:
    for _ in range(173469):
        a = random.random()
        if a < 0.1:
            fp.write("10\n")
        elif a < 0.2:
            fp.write("7\n")
        elif a < 0.3:
            fp.write("9\n")
        elif a < 0.35:
            fp.write("6\n")
        else:
			fp.write("8\n")
