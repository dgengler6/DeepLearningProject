import matplotlib.pyplot as plt 
def mp(ls):
    return list(map(lambda x : x / 10,ls))

# Plots the average error rate for each trial 
def plot_results(base, ws, ws_al, dropout):
    plt.plot(mp(base), label='base')
    plt.plot(mp(ws), label='ws')
    plt.plot(mp(ws_al), label='ws + al')
    plt.plot(mp(dropout), label='Dropout')
    plt.xlabel("Trial Number")
    plt.ylabel("Test Error Rate (%)")
    plt.legend()
    plt.savefig("benchmark_results.png")
    plt.show()
