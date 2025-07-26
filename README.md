🧠 Bigram Language Model with PyTorch 🔥
A simple yet powerful character-level language model inspired by Andrej Karpathy's "nanoGPT" — built from scratch using PyTorch. This project demonstrates how neural networks can learn to generate text one character at a time, using just bigram (2-character) relationships!

🔧 What’s Inside
✅ Clean data preprocessing with character-to-index encoding

✅ Minimal neural network architecture with a trainable embedding table

✅ Custom training loop with best-loss tracking and model checkpointing

✅ Text generation function that continues a prompt character by character

✅ Fully commented, beginner-friendly code for educational use

📘 How It Works
Reads a plain-text dataset and splits it into train/validation sets

Encodes the text into integers (vocabulary of unique characters)

Trains a neural network to predict the next character from the previous one

Generates new text based on a seed like "hello" or a single character

🚀 Perfect For:
Beginners learning PyTorch

Understanding how language models work at a low level

Experimenting with embeddings and token prediction

📂 Files Included:
bigram_dataprep.py – handles encoding, decoding, and batching

bigram_NN.py – builds, trains, and evaluates the bigram neural network
