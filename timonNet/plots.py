import matplotlib.pyplot as plt
import json

files = ['log_twitter_25.json',
         'log_twitter_200.json',
         'log_unsup_25.json',
         'log_unsup_200.json']
         
for file in files:
    with open(file) as openfile:
        data = json.load(openfile)

    plt.plot(data['training']['loss'])
    plt.plot(data['validation']['loss'])
    plt.show()
    
    plt.plot(data['training']['accuracy'])
    plt.plot(data['validation']['accuracy'])
    plt.show()



