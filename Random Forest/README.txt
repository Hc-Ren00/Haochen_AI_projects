To see this code, just simply open the source code named "src.py".

This project applies the random forest algorithm and tests the performance of my model.

If we want to run dataset with numerical attributes, simply use the code on
line 88 and make the code on line 89 as comment. If we want to run dataset 
with categorical attributes, use the code on line 89 and make the code on line
88 as comment. 

Since the file we read may have different format, code at line 16 can be added
or not. This specific code is just to make sure that the variable "line" should
be a list containing each element in a row of the input file.

Also, different file may have different class name, thus we have to hardcode
it at line 303, 337, and 363.

To change the input file, change line 12.

Line 337 is the code where to change the value of minimal size to stop.

From line 394 to line 460, this is where I collect the data and draw graphs.
This is also the place to look at if we want to change the number of folds 
to use.

The performance of my model is shown in those graphs.


  