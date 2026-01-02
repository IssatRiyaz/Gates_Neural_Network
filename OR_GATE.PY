import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Input
X = np.array([[0,0],[0,1],[1,0],[1,1]])

# OR gate output
y = np.array([0,1,1,1])

# Model
model = Sequential([
    Dense(1, activation='sigmoid', input_shape=(2,))
])

model.compile(
    optimizer=Adam(learning_rate=0.01),
    loss='binary_crossentropy'
)

# Train
model.fit(X, y, epochs=400, verbose=0)

# Predict
pred = model.predict(X)
print(pred)
print((pred > 0.5).astype(int))
