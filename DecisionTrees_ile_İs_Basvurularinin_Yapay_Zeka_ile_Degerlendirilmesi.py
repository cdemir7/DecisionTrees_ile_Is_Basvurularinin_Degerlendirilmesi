#Gerekli kütüphanelerimizi içe aktaralım.
import numpy as np
import pandas as pd
from sklearn import tree


#Veri setindeki verileri dataframe'e dönüştürüelim.
df = pd.read_csv("DecisionTreesClassificationDataSet.csv")
#print(df.head())


#Scikit-learn kütüphanesi, desicion tree'lerin düzgün çalışabilmesi için
#veri setinde bulunan bütün verilerin rakamsal olmasını bekler.
#Bundan dolayı veri setindeki "Y" ve "N" değerlerini 0 ve 1 olarak değiştiriyoruz.
duzeltme_mapping = {"Y":1, "N":0}
df["IseAlindi"] = df["IseAlindi"].map(duzeltme_mapping)
df["SuanCalisiyor?"] = df["SuanCalisiyor?"].map(duzeltme_mapping)
df["Top10 Universite?"] = df["Top10 Universite?"].map(duzeltme_mapping)
df["StajBizdeYaptimi?"] = df["StajBizdeYaptimi?"].map(duzeltme_mapping)


#Aynı nedenden ötürü eğitim seviyesindeki verileri de rakamsal hale getiriyoruz.
duzeltme_mapping_egitim = {"BS":0, "MS":1, "PhD":2}
df["Egitim Seviyesi"] = df["Egitim Seviyesi"].map(duzeltme_mapping_egitim)
print(df.head())


#Şimdi sonuç sütununu ayırıyoruz.
y = df["IseAlindi"]
X = df.drop(["IseAlindi"], axis=1)


#Desicion tree modelini oluşturalım.
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)


#Modeli eğittikten sonra şimdi test verisini verelim ve sonucu görelim.
print(clf.predict([[5,1,3,0,0,0]]))   #5 yıl deneyimli, Şu anda çalışan, 3 şirkette çalışmış, Eğitim seviyesi lisans, top mezunu değil, Bizde staj yapmamış
#İşe Alındı: 1
print("---------------------------")
print(clf.predict([[2,0,7,0,1,0]]))   #2 yıl deneyimli, Şu anda çalışmayan, 7 şirkette çalışmış, Eğitim seviyesi lisans, top mezunu , Bizde staj yapmamış
#İşe Alınmadı: 0