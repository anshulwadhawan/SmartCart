import pickle


def clearRecords():
    '''Opens the pickle file and rewrites it with an
    empty dictionary and also resets other variables'''
    pickle_out = open("faces.pickle","wb")
    pickle.dump({}, pickle_out)
    pickle_out.close()


clearRecords()
