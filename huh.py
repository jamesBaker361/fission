with open("metadata.csv","w+") as file:
    file.write("hello wolrd")
    
for k in range(3):
    with open("metadata.csv","a") as file:
        file.write(" hello wolrd "+str(k))