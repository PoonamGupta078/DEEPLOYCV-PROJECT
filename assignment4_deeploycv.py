#Task 1
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
iris=load_iris()
X=iris.data
Y=iris.target
print(X[0])
print(Y[0])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
print(X_train_scaled[0])
print(X_test_scaled[0])
print(X_train.shape)
print(Y_train.shape)
Y_train_encoded=to_categorical(Y_train)
Y_test_encoded=to_categorical(Y_test)
print(Y_train_encoded.shape)
print(Y_test_encoded.shape)
#Task 2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


model=Sequential()
model.add(Dense(8,input_dim=4,activation='relu'))
model.add(Dense(3,activation='softmax'))
#Softmax:-  Softmax function for a set of numbers
# y=e^x(i)/summation of e^x(j) from j=1 to n , summation over all y(i)=1
 #y(i)'s are non-negative and sum to 1, satisfying the properties of a probability distribution.
#2 Softmax transforms raw scores (logits) into probabilities. It emphasizes the largest values and suppresses others, which helps in classification tasks by providing a clear probabilistic interpretation for each possible class.



#Task 3
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
model.fit(X_train_scaled,Y_train_encoded,epochs=100,batch_size=5)
#Task 4
loss, accuracy = model.evaluate(X_test_scaled, Y_test_encoded)

# Step 10: Print the accuracy
print(f"Test Accuracy: {accuracy * 100:.2f}%")



