def test(hello=True):
    if hello:
        print("test")
    else:
        print("test5")
        
def test2(hello=True):
    if hello:
        print("test2")
    else:
        print("test6")
        
def test3(hello=True):
    if hello:
        print("test3")
    else:
        print("test7")
        
def test4(hello=True):
    if hello:
        print("test4")
    else:
        print("test8")
    
test5 = [test,test2,test3,test4]
# test5 = [test,test2,test3,test4,test(False),test2(False),test3(False),test4(False)]

for i in test5:
    i()