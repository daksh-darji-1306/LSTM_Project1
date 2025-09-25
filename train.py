import numpy as np
import requests
import re
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

print("--- Starting the training script ---")

# --- Step 1: Download and Clean the Data ---
print("Step 1: Downloading and cleaning data...")

url = "https://www.gutenberg.org/files/11/11-0.txt"
response = requests.get(url)
text_data = response.text

start_marker = "*** START OF THIS PROJECT GUTENBERG EBOOK ALICE'S ADVENTURES IN WONDERLAND ***"
end_marker = "*** END OF THIS PROJECT GUTENBERG EBOOK ALICE'S ADVENTURES IN WONDERLAND ***"
start_index = text_data.find(start_marker) + len(start_marker)
end_index = text_data.find(end_marker)

story_text = text_data[start_index:end_index]
cleaned_text = story_text.lower()
cleaned_text = re.sub(r'[^a-z\s\']', '', cleaned_text)
cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

print("Data downloaded and cleaned successfully.")

# --- Step 2: Tokenization and Sequence Creation ---
print("Step 2: Tokenizing text and creating sequences...")

tokenizer = Tokenizer()
tokenizer.fit_on_texts([cleaned_text])
word_to_index = tokenizer.word_index
vocab_size = len(word_to_index) + 1

all_tokens = tokenizer.texts_to_sequences([cleaned_text])[0]

sequences = []
sequence_length = 50

for i in range(sequence_length, len(all_tokens)):
    seq = all_tokens[i-sequence_length : i+1]
    sequences.append(seq)

print("Total sequences created: {}".format(len(sequences)))

# --- Step 3: Prepare Data for the Model ---
print("Step 3: Preparing data for the model...")

sequences = np.array(sequences)
X = sequences[:, :-1]
y = sequences[:, -1]
y = to_categorical(y, num_classes=vocab_size)

print("Shape of X (input data): {}".format(X.shape))
print("Shape of y (labels): {}".format(y.shape))

# --- Step 4: Build the LSTM Model ---
print("Step 4: Building the LSTM model...")

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=100, input_length=sequence_length),
    LSTM(150),
    Dense(vocab_size, activation='softmax')
])
model.summary()

# --- Step 5: Compile and Train the Model ---
print("Step 5: Compiling and training the model...")

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X, y, epochs=50, batch_size=128, verbose=2)
print("Model training completed.")

# --- Step 6: Save the Model and Tokenizer ---
print("Step 6: Saving the trained model and tokenizer...")

model.save('next_word_model.h5')

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("--- Assets saved successfully! You are ready to deploy. ---")