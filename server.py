import os.path

import cherrypy

import sqlite3

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors


conn = sqlite3.connect('activities.db', isolation_level=None)
curr = conn.cursor()
curr.execute('select * from act')
acts = curr.fetchall()

df = pd.DataFrame(acts, columns=['act', 'compl', 'price'], dtype=int)
df.index = df['act'].values
df.drop(columns=['act'], inplace=True)

df_t = df.copy()
scaler = MinMaxScaler()
df_t[['compl', 'price']] = scaler.fit_transform(df_t)

model_knn = NearestNeighbors(metric='manhattan', algorithm='brute', n_neighbors=100, n_jobs=-1)
model_knn.fit(df_t)

class WelcomePage:

    @cherrypy.expose
    def index(self):
        # Ask for the user's name.
        return '''
            <form action="greetUser" method="GET">
            What is your name?
            <input type="text" name="name" />
            <input type="submit" />
            </form>'''

    @cherrypy.expose
    def greetUser(self, name=None):
        # CherryPy passes all GET and POST variables as method parameters.
        # It doesn't make a difference where the variables come from, how
        # large their contents are, and so on.
        #
        # You can define default parameter values as usual. In this
        # example, the "name" parameter defaults to None so we can check
        # if a name was actually specified.

        if name:
            # Greet the user!
            return "Hey %s, what's up?" % name
        else:
            if name is None:
                # No name was specified
                return 'Please enter your name <a href="./">here</a>.'
            else:
                return 'No, really, enter your name <a href="./">here</a>.'
            
    @cherrypy.expose
    def getAktivnosti(self):
        conn = sqlite3.connect('activities.db', isolation_level=None)
        curr = conn.cursor()
        curr.execute('select activnost from act')
        acts = curr.fetchall()
        return ",".join([x[0] for x in acts])
    
    @cherrypy.expose
    def sortByParams(self, price, complexity):
        scaled = scaler.transform([[complexity, price]])
        indices = model_knn.kneighbors(scaled, n_neighbors=len(df_t))[1][0]
        return ",".join(df.iloc[indices].index.values)
        

tutconf = os.path.join(os.path.dirname(__file__), 'tutorial.conf')

if __name__ == '__main__':
    # CherryPy always starts with app.root when trying to map request URIs
    # to objects, so we need to mount a request handler root. A request
    # to '/' will be mapped to HelloWorld().index().
    cherrypy.quickstart(WelcomePage(), config=tutconf)