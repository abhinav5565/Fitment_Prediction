from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
l.fit(['paris'])
f=l.transform(['paris'])

g=l.transform(['india'])
print(f)
print(g)
k=l.transform(['paris'])
print(k)