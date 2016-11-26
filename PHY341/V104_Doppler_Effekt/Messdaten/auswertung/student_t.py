import numpy as np

invw, invw_f , m_laut, m_lautf, m_sch, m_sch_f=np.genfromtxt('steigung_der_geraden.txt',unpack=True)

n_w=5
n_f=10
def fehler(m_1f,m_2f,n_1,n_2):
	return np.sqrt((((m_1f**2)*(n_1-1)+(m_2f**2)*(n_2-1))/(n_1+n_2-2))*(n_1+n_2)/(n_1*n_2))

def t(m_1,m_2,F):
	return ((m_1-m_2)/F)


F= fehler(m_sch_f,m_lautf,n_f,n_f)
T= t(m_sch,m_laut,F)

print('T-Faktor: ', T)
print('Abweichung: ', F)