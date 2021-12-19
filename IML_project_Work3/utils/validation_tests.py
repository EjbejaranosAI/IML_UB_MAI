from scipy import stats
x = {'ib1': {'accuracy': 0.8476909230891913, 'variance': 0.001825316549551775, 'time': 0.19239549999999728, 'memory': 1.0}, 'ib2': {'accuracy': 0.7839731937043214, 'variance': 0.001257384654777847, 'time': 0.07404059999999646, 'memory': 0.24928}, 'ib3': {'accuracy': 0.7043506703722858, 'variance': 0.009441290782517167, 'time': 17.400145699999953, 'memory': 0.07392000000000001}}
def dict_to_list(dict):
     ll = []
     items = list(dict.items())
     for i in items:
          l = []
          itms = list(i[1].items())
          for ii in itms:
               l.append(ii[1])
          ll.append(l)
     return ll

def t_test(dict):
     tpvalues = []
     keys = list(dict.keys())
     items = dict_to_list(dict)
     pairs_items = ([(items[i], items[j]) for i in range(len(items)) for j in range(i + 1, len(items))])
     pairs_keys = ([(keys[i], keys[j]) for i in range(len(keys)) for j in range(i + 1, len(keys))])
     for pi, pk in zip(pairs_items, pairs_keys):
          t, p = stats.ttest_ind(pi[0], pi[1])
          tpvalues.append((t,p))
          print(f'For pair {pk[0]} and {pk[1]}, the p-value was {p} and t-value was {t}')
     return tpvalues

def friedman_test(dict):
     items = dict_to_list(dict)
     print(items[0])
     return stats.friedmanchisquare(items[0], items[1])

