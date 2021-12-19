from scipy import stats

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
