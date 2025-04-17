import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Tahap 1. Memuat dan membaca dataset yang sudah di download lalu memahami pola awal dari data
# Load data
df = pd.read_csv('Lung Cancer Dataset.csv')

# Tampilkan 5 data teratas
print(df.head())

# Info data
print(df.info())

# Statistik deskriptif
print(df.describe())

# Visualisasi distribusi umur
sns.histplot(df['AGE'], bins=10, kde=True)
plt.title('Distribusi Umur Pasien')
plt.show()


# Tahap 2. mengecek dan menangani data kosong yang duplikat

print("Missing values per kolom:")
print(df.isnull().sum())

# Cek duplikat
print("\nJumlah duplikat data:", df.duplicated().sum())

# (Opsional) Hapus duplikat jika ada
df = df.drop_duplicates()




# Tahap 3. Konversi label 
# Konversi label YES/NO jadi 1/0
df.replace({'YES': 1, 'NO': 0, 'M': 1, 'F': 0}, inplace=True)

# Cek hasil konversi
print(df.head())





# Tahap 4. Visualisasi distribusi dan korelasi

plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Heatmap Korelasi antar Fitur')
plt.show()

# Distribusi GENDER terhadap kanker
sns.countplot(x='GENDER', hue='LUNG_CANCER', data=df)
plt.title('Kanker Paru-paru berdasarkan Gender')
plt.show()

# Distribusi SMOKING terhadap kanker
sns.countplot(x='SMOKING', hue='LUNG_CANCER', data=df)
plt.title('Kanker Paru-paru dan Kebiasaan Merokok')
plt.show()





# Tahap 4. Filter kategori berdasarkan umur

df['AGE_GROUP'] = pd.cut(df['AGE'], bins=[0, 30, 45, 60, 100],
                         labels=['<30', '30-45', '45-60', '60+'])

# Visualisasi distribusi kanker per kelompok umur
sns.countplot(x='AGE_GROUP', hue='LUNG_CANCER', data=df)
plt.title('Distribusi Kanker berdasarkan Kelompok Usia')
plt.show()




