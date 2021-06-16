import os 

cwd = os.getcwd() # gives directory from where program is called! 
dirname = os.path.dirname(__file__) # gives parent dir of file 
print(cwd, "\n",dirname)