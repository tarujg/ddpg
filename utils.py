import matplotlib.pyplot as plt

def plotting(parameter, title, plotname, xlabel, ylabel):

    plt.figure()
    plt.plot(parameter)    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(plotname)
    plt.show()