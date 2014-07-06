z = [0,0,0,0,0]
a = 'print z[1]&z[3]'

e = compile(a,'an','exec') 
exec(e)

on = [1,3]
for i in on :
    z[i]=1


exec(e)
 
for i in on :
    z[i]=0

exec(e)


