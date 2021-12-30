from bs4 import BeautifulSoup
soup=BeautifulSoup('aa','lxml')
print(soup.p.string)
import lxml
import sqlite3
import os


path = os.path.dirname(__file__)

con = sqlite3.connect(path+'/example.db')
cur = con.cursor()

# Create table
rows=cur.execute('select * from stock')

for row in rows:
    print(row)

# We can also close the connection if we are done with it.
# Just be sure any changes have been committed or they will be lost.
con.close()