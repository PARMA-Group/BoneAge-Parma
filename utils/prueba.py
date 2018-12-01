import _pickle as pickle

def read_content(name):
    pickle_off = open(name,"rb")
    emp = pickle.load(pickle_off)
    lista = list(emp.keys())
    print(lista)
    print(len(emp[lista[0]]))
    print(len(emp[lista[1]]))
    pickle_off.close()

a = [   "dnlm0f_prediction_results.p", 
        "dnlm0m_prediction_results.p", 
        "nodnlm0f_prediction_results.p",
        "nodnlm0m_prediction_results.p"]


read_content("training_results.p")
b = []

for i in b:
    pickle_off = open(i,"rb")
    emp = pickle.load(pickle_off)
    lista = list(emp.keys())
    print(lista)
    print(emp[lista[0]])
    print(emp[lista[1]])
    pickle_off.close()