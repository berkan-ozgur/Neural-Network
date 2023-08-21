import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Örnek veri seti oluşturumu
data = np.random.random((1000, 20))  # 1000 örnek, her biri 20 özellik içerir
labels = to_categorical(np.random.randint(10, size=(1000,)))  # 10 sınıf

# Veri setini eğitim ve test kümelerine ayırma
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2)

# Yapay sinir ağı modelinin oluşturulması
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=20))
model.add(Dense(64, activation='relu'))
# Çıkış katmanı, sınıf sayısı kadar nöron ve softmax aktivasyonu kullanır
model.add(Dense(10, activation='softmax'))

# Modelin derlenmesi
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# Modelin eğitimi
model.fit(X_train, y_train, epochs=10, batch_size=32,
          validation_data=(X_test, y_test))

# Modelin değerlendirilmesi
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")
