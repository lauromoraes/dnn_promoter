class MyDatabase(object):

    def setDB(self, ppath, npath):
        import pandas as pd
        self.pos = pd.read_csv(ppath, header=None)
        self.neg = pd.read_csv(npath, header=None)


    def __init__(self):
        print('OK')

db = MyDatabase()
db.setDB('convDiPos.csv', 'convDiNeg.csv')