import matlab.engine
eng = matlab.engine.start_matlab()
# tf = eng.triarea(nargout=0) # Prints no output
Y =  matlab.double([5.0,4.0,3.0,2.0,1.0])
tf = eng.test(Y)
print(tf)