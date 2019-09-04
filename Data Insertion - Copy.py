import os
from pathlib2 import Path
from bs4 import BeautifulSoup
import mysql
import mysql.connector

rootDir = 'C:\Phishing\data'
iter1=0
#!/usr/bin/python

cnx = mysql.connector.connect(user='root', password='',
                              host='127.0.0.1',
                              database='new_schema')
print(cnx)

cursor = cnx.cursor()
 
# One time Code for Data insertion in MySql.
for dirName, subdirList, fileList in os.walk(rootDir):
    print('Found directory: %s' % dirName)
    for fname in fileList:
        print('\t%s' % dirName+fname)
        iter1=iter1+1        
        try:
            contents = Path(dirName+'\\'+fname).read_text()
            soup = BeautifulSoup(contents,'html.parser')
            print(iter1)
            message=soup.prettify()
                  
            if(str(dirName+fname).count("spam")):
                print("Spam Email")
                spamindicator="1"
                indicator="Spam"
            else:
                print("Ham Email")
                spamindicator="0"
                indicator="Ham"
                
            sql = "INSERT INTO new_table3 VALUES('"+str(iter1)+"','"+cnx.converter.escape(message)+"','"+spamindicator+"','"+indicator+"')"
            cursor.execute(sql)
            cnx.commit()
            print("success")
            #except:
                    #print("Unexpected error:")
        except:
            print("Error")





