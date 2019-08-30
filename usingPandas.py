import pandas as pd
from IPython.display import display
data = {'First Name': ["Frodo", "Bilbo", "Aragorn II", "Samwise"],
        'Last Name': ["Baggins", "Baggins", "Elessar", "Gamgee"],
        'Place of birth': ["Shire", "Shire", "Eriador", "Shire"],
        'Date of Birth T.A.': [2968, 2890, 2931, 2980]
        }
data_pandas = pd.DataFrame(data)
display(data_pandas)

data_pandas = pd.DataFrame(data, index=['Frodo','Bilbo','Aragorn','Sam'])
display(data_pandas)

display(data_pandas.loc['Aragorn'])

new_hobbit = {'First Name': ["Peregrin"],
              'Last Name': ["Took"],
              'Place of birth': ["Shire"],
              'Date of Birth T.A': [2990]
              }
data_pandas = data_pandas.append(pd.DataFrame(new_hobbit, index=['Pippin']))
display(data_pandas)
