import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Embedding, LSTM, Dense

model=keras.Sequential([
    Embedding(input_dim=10000,output_dim=32,input_length=100),
    LSTM(64),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
test_loss, test_acc= model.evaluate(x_test,y_test,verbose=2)
print(f'Точность на тестовых данных:{test_acc}')
