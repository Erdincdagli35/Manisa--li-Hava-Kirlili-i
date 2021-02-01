# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 16:32:00 2020

@author: Erdinc
"""

#%%Libaries 
import numpy as np 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

#%%datasetin projeye dahil edilmesi ve bilgilerinin gösterilmesi -- Veri Manipülasyonu Bölümü
def readDatasetNinformantions():    
    product=pd.read_excel("data.xlsx")
    product.head()
    df = product.copy()
    df.head()
    #data information
    df.info()
    df.dtypes
    #Numrical details
    df.describe().T
    return df

#%%datasetimizin özeti belirleyebilmek için bir fonksiyon yazacağız
def summaryofDataset():
    df=readDatasetNinformantions()
    print("PM10 ( µg/m³ ) : Avg : "+str(df["PM10 ( µg/m³ )"].mean()))
    print("PM10 Debi ( m³/saat ) : Avg : "+str(df["PM10 Debi ( m³/saat )"].mean()))
    print("SO2 ( µg/m³ ) : Avg : "+str(df["SO2 ( µg/m³ )"].mean()))
    print("CO ( µg/m³ ) : Avg : "+str(df["CO ( µg/m³ )"].mean()))
    print("Hava Sıcaklığı ( °C ) göre dağılımlar : "+str(df["Hava Sıcaklığı ( °C )"].value_counts())) 


#%%Örnek Teorisi - Veri Bilimi ve İstatislik Bölümü 
#Rastgele seçilen bir sütunda nasıl bir ortalama çıkaracağı sonucunu öğrenmek için 
#Örneklem teorisine başvurduk.
def theoryofExampleforSeedPrice():
    df=readDatasetNinformantions()
    np.random.seed(10) 
    example = np.random.choice(a=df["Hava Sıcaklığı ( °C )"],size=50)
    example2 = np.random.choice(a=df["Hava Sıcaklığı ( °C )"],size=50)
    example3 = np.random.choice(a=df["Hava Sıcaklığı ( °C )"],size=50)
    example4 = np.random.choice(a=df["Hava Sıcaklığı ( °C )"],size=50)
    example5 = np.random.choice(a=df["Hava Sıcaklığı ( °C )"],size=50)
    example6 = np.random.choice(a=df["Hava Sıcaklığı ( °C )"],size=50)
    example7 = np.random.choice(a=df["Hava Sıcaklığı ( °C )"],size=50)
    newValue = (example.mean()+example2.mean()+example3.mean()+example4.mean()+example5.mean()+example6.mean()+example7.mean())/6
    print(newValue)
    print(df.mean())
    return newValue

#%%Bernoulli Kavramını uygulama
from scipy.stats import bernoulli
def bernolliCase():
    df=readDatasetNinformantions()
    p= df["Hava Sıcaklığı ( °C )"].min()
    #p = 0.6 
    r = bernoulli(p)
    r.pmf(k=0)  
    return r

#%%Karar Aralığı
import statsmodels.stats.api as sms
def believeCase():
    df=readDatasetNinformantions()
    sms.DescrStatsW(df["Hava Sıcaklığı ( °C )"]).tconfint_mean()
    return df["Hava Sıcaklığı ( °C )"].mean()

#%%Binom Kavramı
from scipy.stats import binom
def binomCase():
    df=readDatasetNinformantions()
    p=df["PM10 Debi ( m³/saat )"]
    n=100
    rv=binom(n,p)
    rv.pmf(5)
    return rv

#%%Possion Kavramı 
from scipy.stats import poisson
def possionCase():
    df=readDatasetNinformantions()
    lambda_=df["PM10 Debi ( m³/saat )"]
    rv=poisson(mu = lambda_)
    rv.pmf(k=0)
    return rv

#%%Veri Önişleme - Aykırı Gözlem
def preProc():
    df=readDatasetNinformantions()
    dataTable = df["O3 ( µg/m³ )"].copy()
    #sns.boxplot(x = dataTable)
    q1 = dataTable.quantile(0.25)
    q3 = dataTable.quantile(0.75)
    iqr =q3-q1
    
    subLine = q1 - 1.5*iqr
    subLine 
    topLine = q3 + 1.5*iqr
    topLine 
    
    (dataTable <(subLine)) | (dataTable > (topLine))
    (dataTable <(subLine))
    cObservation = dataTable<subLine
    cObservation.head(10)
    contradictory=dataTable[cObservation]
    contradictory.index
    #contradictory is empty.i dont have any contradictory
    dataTable = df["O3 ( µg/m³ )"].copy()
    #sns.boxplot(x = dataTable)
    q1 = dataTable.quantile(0.25)
    q3 = dataTable.quantile(0.75)
    iqr =q3-q1
    
    subLine = q1 - 1.5*iqr
    subLine 
    topLine = q3 + 1.5*iqr
    topLine 
    
    (dataTable <(subLine)) | (dataTable > (topLine))
    (dataTable <(subLine))
    cObservation = dataTable<subLine
    cObservation.head(10)
    contradictory=dataTable[cObservation]
    contradictory.index
    #contradictory is empty.i dont have any contradictory
    return dataTable

#%%Çok değişkenli aykırı gözlem
from sklearn.neighbors import LocalOutlierFactor
def preProc2():
    dataTable=preProc()
    X = np.r_[dataTable]
    
    LOF = LocalOutlierFactor(n_neighbors = 20 , contamination = 0.1)
    LOF.fit_predict(X)
    X_score = LOF.negative_outlier_factor_
    return X_score
    
#%%empty area for category veriable 
def checkEmptyArea():
    df=readDatasetNinformantions()
    #Is any value empty ? 
    if df.isnull().values.any() == True:
         df=df.dropna()
         return df
    else:
        return df

#%%Koalasyon grafiği
def cGraph():
    df=checkEmptyArea()
    listFeature = ["Tarih",'PM10 ( µg/m³ )', 'PM10 Debi ( m³/saat )', 'SO2 ( µg/m³ )',
       'CO ( µg/m³ )', 'NO2 ( µg/m³ )', 'NOX ( µg/m³ )', 'NO ( µg/m³ )',
       'O3 ( µg/m³ )', 'Hava Sıcaklığı ( °C )', 'Rüzgar Hızı ( m/s )',
       'Bağıl Nem ( % )', 'Hava Basıncı ( mbar )', 'Ruzgar Yönü ( Derece )']
    sns.heatmap(df[listFeature].corr(), annot = False , fmt = ".2f")
    plt.show()

#%%Feature'larımızı oluşturacağız 
"""
PM10 (24hr)	+
PM2.5 (24hr) -	
NO2 (24hr)	+
O3 (8hr)	+
CO (8hr)	+
SO2 (24hr)	+
NH3 (24hr)	-
Pb (24hr)   +
"""
#    df.loc[df['sonuç'] > 0, ['binarySonuc']] = 1
#df["mazotGider"][df.ÜretimSüresi=="Uzun Süre "]=df["Mazot"][df.ÜretimSüresi=="Uzun Süre "]*M2*20

def addFeature():
    df=checkEmptyArea()
    df["statusPm10"]=0
    df['statusPm10'] = pd.to_numeric(df['statusPm10'], errors='coerce')     
    df.loc[(df["PM10 ( µg/m³ )"]<50) & (df["PM10 ( µg/m³ )"]>=0),['statusPm10']]=5
    df.loc[(df["PM10 ( µg/m³ )"]<101) & (df["PM10 ( µg/m³ )"]>=50),['statusPm10']]=4
    df.loc[(df["PM10 ( µg/m³ )"]<251) & (df["PM10 ( µg/m³ )"]>=100),['statusPm10']]=3
    df.loc[(df["PM10 ( µg/m³ )"]<351) & (df["PM10 ( µg/m³ )"]>=250),['statusPm10']]=2
    df.loc[(df["PM10 ( µg/m³ )"]<431) & (df["PM10 ( µg/m³ )"]>=350),['statusPm10']]=1
    df.loc[(df["PM10 ( µg/m³ )"]>430) ,['statusPm10']]=0
    
    df["statusNo2"]=0
    df['statusNo2'] = pd.to_numeric(df['statusNo2'], errors='coerce') 
    df.loc[(df["NO2 ( µg/m³ )"]<40) & (df["NO2 ( µg/m³ )"]>=0),['statusNo2']]=5
    df.loc[(df["NO2 ( µg/m³ )"]<81) & (df["NO2 ( µg/m³ )"]>=40),['statusNo2']]=4
    df.loc[(df["NO2 ( µg/m³ )"]<181) & (df["NO2 ( µg/m³ )"]>=80),['statusNo2']]=3
    df.loc[(df["NO2 ( µg/m³ )"]<281) & (df["NO2 ( µg/m³ )"]>=180),['statusNo2']]=2
    df.loc[(df["NO2 ( µg/m³ )"]<401) & (df["NO2 ( µg/m³ )"]>=280),['statusNo2']]=1
    df.loc[(df["NO2 ( µg/m³ )"]>400),['statusNo2']]=0
    
    df["statusO3"]=0
    df['statusO3'] = pd.to_numeric(df['statusO3'], errors='coerce') 
    df.loc[(df["O3 ( µg/m³ )"]<50) & (df["O3 ( µg/m³ )"]>=0),['statusO3']]=5
    df.loc[(df["O3 ( µg/m³ )"]<101) & (df["O3 ( µg/m³ )"]>=50),['statusO3']]=4
    df.loc[(df["O3 ( µg/m³ )"]<169) & (df["O3 ( µg/m³ )"]>=100),['statusO3']]=3
    df.loc[(df["O3 ( µg/m³ )"]<209) & (df["O3 ( µg/m³ )"]>=168),['statusO3']]=2
    df.loc[(df["O3 ( µg/m³ )"]<749) & (df["O3 ( µg/m³ )"]>=208),['statusO3']]=1
    df.loc[(df["O3 ( µg/m³ )"]>=748),['statusO3']]=0
    
    df["statusCO"]=0
    df['statusCO'] = pd.to_numeric(df['statusCO'], errors='coerce') 
    df.loc[(df["CO ( µg/m³ )"]<1.1) & (df["CO ( µg/m³ )"]>=0),['statusCO']]=5
    df.loc[(df["CO ( µg/m³ )"]<2.1) & (df["CO ( µg/m³ )"]>=1.1),['statusCO']]=4
    df.loc[(df["CO ( µg/m³ )"]<11) & (df["CO ( µg/m³ )"]>=2.0),['statusCO']]=3
    df.loc[(df["CO ( µg/m³ )"]<18) & (df["CO ( µg/m³ )"]>=10),['statusCO']]=2
    df.loc[(df["CO ( µg/m³ )"]<35) & (df["CO ( µg/m³ )"]>=18),['statusCO']]=1
    df.loc[(df["CO ( µg/m³ )"]>=35) ,['statusCO']]=0
    
    df["statusSO2"]=0
    df['statusSO2'] = pd.to_numeric(df['statusSO2'], errors='coerce') 
    df.loc[(df["SO2 ( µg/m³ )"]<41) & (df["SO2 ( µg/m³ )"]>=0),['statusSO2']]=5
    df.loc[(df["SO2 ( µg/m³ )"]<81) & (df["SO2 ( µg/m³ )"]>=40),['statusSO2']]=4
    df.loc[(df["SO2 ( µg/m³ )"]<381) & (df["SO2 ( µg/m³ )"]>=80),['statusSO2']]=3
    df.loc[(df["SO2 ( µg/m³ )"]<801) & (df["SO2 ( µg/m³ )"]>=380),['statusSO2']]=2
    df.loc[(df["SO2 ( µg/m³ )"]<1601) & (df["SO2 ( µg/m³ )"]>=800),['statusSO2']]=1
    df.loc[(df["SO2 ( µg/m³ )"]>=1600),['statusSO2']]=0
    
    df["ortalamaDurum"]=(df['statusSO2']+df["statusCO"]+df["statusO3"]+df["statusNo2"]+df["statusPm10"])/5
    df["binarySonuc"]=0
    df['binarySonuc'] = pd.to_numeric(df['binarySonuc'], errors='coerce')
    df.loc[df['ortalamaDurum'] > 3, ['binarySonuc']] = 1
    df.binarySonuc
    return df

#%%Korelasyon grafiği
def cGraphAgain():
    listFeature = df.columns
    sns.heatmap(df[listFeature].corr(), annot = True , fmt = ".2f")
    plt.show()

#%%
from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix 
from sklearn.preprocessing import StandardScaler, RobustScaler
def modelling():

    #df=makeStandartion()
    df = addFeature()
    label_encoder =LabelEncoder().fit(df.binarySonuc)
    labels = label_encoder.transform(df.binarySonuc)
    classes = list(label_encoder.classes_)
    df=df.drop(["binarySonuc","Tarih"],axis=1)
    y =df
    X=labels
    nb_features = 20
    nb_classes = len(classes)
    
    from sklearn.utils import shuffle
    X, y = shuffle(X, y)
    
    """
    #Normalization
    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler().fit(df.binarySonuc)
    X=scaler.fit_transform(df.binarySonuc)
    """
    #Variladtion 
    from sklearn.model_selection import train_test_split
    X_train,X_valid,y_train,y_valid=train_test_split(df, labels, test_size=0.3 )
    #Category Part
    from tensorflow.keras.utils import to_categorical
    y_train=to_categorical(y_train)
    y_valid=to_categorical(y_valid)
    
    X_train = np.array(X_train).reshape(2552, 20,1)
    X_valid = np.array(X_valid).reshape(1095, 20,1)
    
    #Model
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense,Activation,SimpleRNN,Dropout,MaxPooling1D,Flatten,BatchNormalization,Conv1D
    import tensorflow as tf
    
    model = Sequential()
    model.add(Conv1D(512,1,input_shape=(nb_features,1)))
    model.add(Activation("relu"))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(256,1))
    model.add(Activation("relu"))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(2048,activation="relu"))
    model.add(Dense(1024,activation="relu"))
    model.add(Dense(nb_classes,activation="sigmoid"))
    model.summary()
    
    
    from keras import backend as K
    
    def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
    
    def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
    def f1_m(y_true, y_pred):
        precision = precision_m(y_true, y_pred)
        recall = recall_m(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall+K.epsilon()))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy',f1_m,precision_m,recall_m])
    
    
    model.fit(X_train,y_train,epochs=100,validation_data=(X_valid,y_valid))
    return model

"""
accuracy = model.score(X_valid, y_valid) 
print(accuracy) # 0.92
 
predictions = model.predict(X_test)  
# creating a confusion matrix 
cm = confusion_matrix(y_valid, predictions) 
"""
#%%
def staticsOfModelling():
    model=modelling()
    print("Ortalama Eğitim Kaybı : ",np.mean(model.history.history["loss"]))
    print("Ortalama Eğitim Başarımı : ",np.mean(model.history.history["accuracy"]))
    print("Ortalama Doğrulama Kaybı : ",np.mean(model.history.history["val_loss"]))
    print("Ortalama Doğrulama Başarımı : ",np.mean(model.history.history["val_accuracy"]))
    print("Ortalama F1 - Skor Değeri : ",np.mean(model.history.history["val_f1_m"]))
    print("Ortalama Kesinlik Değeri : ",np.mean(model.history.history["val_precision_m"]))
    
    
    
    plt.plot(model.history.history["accuracy"])
    plt.plot(model.history.history["val_accuracy"])
    plt.title("Model Başarımları")
    plt.ylabel("Başarım")
    plt.xlabel("Epok")
    plt.legend(["Eğitim","Test"],loc="upper_left")
    plt.show()
    """
    plt.plot(model.history.history["loss"])
    plt.plot(model.history.history["val_loss"])
    plt.title("Model Kayıpları")
    plt.ylabel("Kayıp")
    plt.xlabel("Epok")
    plt.legend(["Eğitim","Test"],loc="upper_left")
    plt.show()
    
    plt.plot(model.history.history["precision_m"],color="g")
    plt.plot(model.history.history["val_precision_m"],color="r")
    plt.title("Model Hasasiyeti Skorları")
    plt.ylabel("Hasasiyeti Skorları")
    plt.xlabel("Epok Sayısı")
    plt.legend(["Eğitim","Doğrulama"],loc="uppper left")
    plt.show
    """
    
#%%Main Fonksiyonu 
def main():
    #Veri Analizi
    print ("--Veri Analizi Bölümü-- \n\n\n")
    summaryofDataset()
    
    #Veri Bilimi ve İstatistik Bölümü
    print ("--Veri Bilimi ve İstatistik Bölümü-- \n\n\n")  
    theoryofExampleforSeedPrice()
    #bernolliCase()
    believeCase()
    binomCase()
    possionCase()
    
    #Veri Önişleme
    print ("--Veri Önişleme Bölümü-- \n\n\n") 
    preProc()
    #preProc2()
    checkEmptyArea()       
    #cGraph()    
    
    #Veri Kümesi için Öznitelik Belirleme
    print ("--Veri Kümesi için Öznitelik Belirleme-- \n\n\n") 
    addFeature()
    #editDataSetforSQLite()
    #cGraphAgain()
    #makeDummies()
    
    #Veri Kümesi Modellenmesi
    print ("--Veri Kümesi Modellenmesi-- \n\n\n") 
    modelling()
    
    #Modelleme Sonuçları
    print ("--Model Sonuçları-- \n\n\n")
    staticsOfModelling()
    
#%%Main kısmını çalıştır
if __name__ == '__main__':
    #Veri Manipülasyonu Bölümü
    print ("--Veri Manipülasyonu Bölümü-- \n\n\n")
    df=readDatasetNinformantions()  
    
    main()
