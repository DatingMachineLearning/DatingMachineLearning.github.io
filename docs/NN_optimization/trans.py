from os import listdir

if __name__ == '__main__':
    all = []
    for i in listdir(r"C:\Users\kevty\Documents\file backups\Dating Machine Learning\docs\NN_optimization"):
        item = "NN_optimization/" + i
        print("- [{}]({})".format(i[:-3], item))