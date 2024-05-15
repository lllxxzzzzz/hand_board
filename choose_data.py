
import os
import shutil
dir=os.listdir(r'C:\Users\24223\Desktop\hand\labels')
for d in dir:
    die=r'C:\Users\24223\Desktop\hand\data\\'+d[:-3]+'jpg'
    print(die)
    if os.path.exists(die):
        print(True)
        shutil.copy(die,
                    r'C:\Users\24223\Desktop\hand\images')
for d in dir:
    die=r'C:\Users\24223\Desktop\hand\data\\'+d[:-3]+'png'
    print(die)
    if os.path.exists(die):
        print(True)
        shutil.copy(die,
                    r'C:\Users\24223\Desktop\hand\images')