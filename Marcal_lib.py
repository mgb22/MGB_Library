"""
Llibreria d'Algorismia i Programacio Audiovisual
Terrassa - ESEIAAT
Marcal Garcia Boris
Maig del 2020
"""
import struct
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as scs

"""
----------------------------------WAV------------------------------------------
"""
def lectura_WAV(fitxer):
    """
    Reading a .file.
    """
    f = open(fitxer,mode='rb')
    fmt = '<24si12si'
    no_1,fm,no_2,n_mostres = struct.unpack(fmt,f.read(struct.calcsize(fmt)))
    n_mostres//=2
    fmt2 = '<{}h'.format(n_mostres)
    cos = list(struct.unpack(fmt2,f.read(struct.calcsize(fmt2))))
    f.close()
    return fm,cos

def escriptura_WAV(fitxer,freq_mostratge,senyal_audio):
    """
    Writing a .wav file using the sampling frequency, the audio signal and the new file's name
    Only works with mono signals, 16-bit per sample
    """
    fic_Out = open(fitxer,mode='wb')
    
    if np.max(np.abs(senyal_audio)) >= 32767:   #Rescale the signal
        senyal_audio = (senyal_audio/np.abs(senyal_audio).max()*(2**(15)-1)).astype(int)
    
    n_mostres = 0
    for i in range(0,len(senyal_audio)):
        n_mostres += 2
    
    mostres_totals = n_mostres + 36
    bits_mostra = 16
    bytes_segon = (freq_mostratge*1*bits_mostra)//8 #Calculation of mono signal
    
    header = [1179011410,mostres_totals,1163280727,544501094,
              16,1,1,freq_mostratge,
              bytes_segon,2,bits_mostra,1635017060,
              n_mostres]
    print(header)
    
    fmrh = '<5i2h2i2h2i' #Format to write all the header
    fic_Out.write(struct.pack(fmrh,*header))
    fmrd = '<{}h'.format(n_mostres//2) #Format to write the data
    fic_Out.write(struct.pack(fmrd,*senyal_audio))
    
def delmat(senyal_audio,D):
    """
    Split the signal into tens from a factor D
    """
    return senyal_audio[0::D] 

def interpolat(senyal_audio,I): 
    """
    Interpolate the signal from a factor I
    """
    contador = 0
    auxiliar = senyal_audio.copy()
    for i in range (0,(len(auxiliar)+(len(auxiliar)//2)*I)): #Final quantity
        if contador == 2:
            contador=-I+1
            for j in range(0,I):
                auxiliar.insert(i,0)
        else:
            contador += 1
    return auxiliar

"""
------------------------------------TF-----------------------------------------
Fourier's transform
"""
def transformadaFourier(x,Nf):
    """
    Discreet Fourier's Transform
    """
    if Nf % 2:
        N = 2 * Nf - 1
    else:
        N = 2 * (Nf-1)
    Lx = len(x)
    if Lx > N:
        Lx = N
    X = np.zeros(Nf) + 0j
    x = x[0:Lx]
    for k in range (Nf):
        ec = np.zeros(Lx) + 0j
        for n in range (Lx):
            ec[n] = np.exp(-2j*np.pi*k*n/N) 
        X[k] = np.array(x)@np.array(ec) 
    return X,N

def transformadaFourier_F1(x,Fx):
    """
    Discreet Fourier's Transform for one frequency
    """
    Lx = len(x)
    n = np.arange(Lx)
    X = (np.exp(-2j*np.pi*Fx*n))@x
    return X

"""
--------------------------------FIR Filter-------------------------------------
FIR Filter
"""
class FiltreFIR:
    '''
    Order M = L - 1
    '''
    def __init__(self,b,v=[]):
        self.b = b
        self.L = len(b)
        self.v = v
        if len(v) != self.L: self.v=np.zeros(self.L-1)
    
    def __reset__(self):
        self.v = np.zeros(self.L-1)
        
    def __call__(self,x):
        Lx = len(x)
        M = len(self.v)
        y = np.zeros(Lx)
        for n in range (Lx):
            y[n] = x[n]*self.b[0] + self.b[1:]@self.v
            self.v[1:]=self.v[0:M-1]
            self.v[0] = x[n]
        return y
    
    def __repr__(self):
        return f"FIR Filter ({self.b} \n {self.v})"
    
"""
------------------------------Ideal FIR Filter---------------------------------
"""
def fir_optim(Fp,ap,Fa,aa):
    Fideal=[0,Fp,Fa,0.5]
    Hmideal=[1,0]
    delta_a=10**(-aa/20)
    delta_p=(10**(ap/20)-1)/(10**(ap/20)+1)
    contrapes=[delta_a,delta_p]
    
    limit = 1-delta_p    
    ideal = 0
    
    for L in range(2,100):
        b=scs.remez(L,Fideal,Hmideal,contrapes)
        B=transformadaFourier_F1(b,Fp)
        if np.abs(B) > limit:
            ideal = L
            break
    b_ideal = scs.remez(ideal,Fideal,Hmideal,contrapes)
    return b_ideal
   
"""
--------------------------------IIR Filter-------------------------------------
"""    
class FiltreIIR:
    '''
    IIR Filter supposing order Q = M
    '''
    def __init__(self,a,b,v=[]):
        self.a = a
        self.b = b
        self.L = len(self.b)
        self.v = v
        if len(v) != self.L: self.v=np.zeros(self.L-1)
        
    def __reset__(self):
        self.v = np.zeros(self.L-1)
    
    def __call__(self,x):
        Lx = len(x)
        M = len(self.v)
        y = np.zeros(Lx)
        for n in range(Lx):
            y[n] = self.b[0]*x[n] + self.v[0]
            for i in range(1,M):
                self.v[i-1] = self.b[i]*x[n] - self.a[i]*y[n] + self.v[i]
            self.v[M-1] = self.b[M]*x[n] - self.a[M]*y[n]
        return y
    
    def __repr__(self):
        return f"IIR Filter ({self.a} \n {self.b} \n {self.v})"
    
    def __str__(self):
        return f"IIR Filter with order M={self.L-1} \n L={self.L} coeficients"
    
    def __help__(self):
        return f"Allocation: nom_filtre=FiltreIIR(a,b,v) \n Initialize internal state: nom_filtre.__reset__() \n Filtration of signal 'x'': y = nom_filtre(x)"

"""
-------------------------------Poles and Zeros------------------------------------
"""
def pols_zeros(b,a=[1]):
    zeros = np.roots(b)
    pols = np.roots(a)
    F_n = np.arange(720)/720
    cir_uni = np.exp(2j*np.pi*F_n)
    plt.title('Poles and zeros')
    plt.plot(np.real(zeros),np.imag(zeros),'og',label='zeros')
    plt.plot(np.real(pols),np.imag(pols),'xr',label='poles')
    plt.plot(np.real(cir_uni),np.imag(cir_uni),':b')
    plt.legend()
    plt.axis('square')
    plt.grid()
    
"""
-----------------------------------Module--------------------------------------
"""
def representacio_modul(Fp,ap,Fa,aa):
    """
    -------------------------------RUNNING-------------------------------------
   representacio_modul() represents with horizontal and vertical lines the behavior
   of one filter considering its design.
    """
    delta_p=(10**(ap/20)-1)/(10**(ap/20)+1)
    delta_a=(10**(-aa/20))
    plt.plot([0,Fp],[1+delta_p,1+delta_p],c='r')
    plt.plot([0,Fp],[1-delta_p,1-delta_p],c='r')
    plt.plot([Fp,Fp],[0,1-delta_p],c='r')
    plt.plot([Fa,Fa],[delta_a,1-delta_p],c='r')
    plt.plot([Fa,0.5],[delta_a,delta_a],c='r')
    
"""
----------------------------------Gain-----------------------------------------
"""
def representacio_guany(Fp,ap,Fa,aa):
    """
    ----------------------------RUNNING-----------------------------------
    representacio_guany() represents with horizontal and vertical lines the
    filter's gain according its design
    """
    delta_p=(10**(ap/20)-1)/(10**(ap/20)+1)
    gp1=20*np.log(1+delta_p)
    gp2=20*np.log(1-delta_p)
    ap=gp1-gp2
    ga=-aa
    plt.plot([0,Fp],[gp1,gp1],c='r')
    plt.plot([0,Fp],[gp2,gp2],c='r')
    plt.plot([Fp,Fp],[ga,gp2],c='r')
    plt.plot([Fa,Fa],[0,ga],c='r')
    plt.plot([Fa,0.5],[ga,ga],c='r')















