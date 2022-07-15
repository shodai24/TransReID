import graphs
import os

statter = graphs.TrainStat(r"../misc/Final-Models/Approach-1/VeRi/")
statter.load_train_log(r"../misc/Final-Models/Approach-1/VeRi/train_log.txt")
statter.plot()
#statter.show()
statter.save(r"../misc/Final-Models/Approach-1/VeRi/")