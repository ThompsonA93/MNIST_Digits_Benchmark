# Common configurations for other python/jupyter files
# For repeated iteration, please refer to the SHELL (.sh) or POWERSHELL files (.ps1)

import platform

# Training-Size
# 60000 for full data set
# 10000 for full data set
num_train = 15000                    
num_test  = 2500                   
                                  
# Use GridSearchCV to look up optimal parameters - Separate from actual training; takes a long time.
# True/False: Run hyper-parameter search via GridSearchCV. 
hyper_parameter_search = True      

# For echo operating system parameters
os = platform.platform()
cpu = platform.processor()
