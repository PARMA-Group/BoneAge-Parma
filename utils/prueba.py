import _pickle as pickle

for i in ["dnlm0f_prediction_results.p", "dnlm0m_prediction_results.p", "nodnlm0f_prediction_results.p", "nodnlm0m_prediction_results.p"]:
    pickle_off = open(i,"rb")
    emp = pickle.load(pickle_off)
    lista = list(emp.keys())
    print(lista)
    print(emp[lista[0]])
    print(emp[lista[1]])
    pickle_off.close()