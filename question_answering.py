from googlesearch import search
query =input("enter your question")
for i in search (query,tld="com" ,num=5,stop=5,pause=2):
    print(i)
