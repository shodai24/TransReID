import graphs

statter = graphs.TrainStat()
statter.load_train_log("..\\misc\\08022022\\train_log.txt")
statter.plot()
statter.show()