# SIMPLE_RNN

### **Key Idea: Why Order Matters Differently in ANN vs. RNN**  
In **ANN**, order matters **only for the input features** because the positions of those features are fixed by you. For example:  

| Price | Size |  
|-------|------|  
| 100   | 200  |  

If you reverse the order:  
| Size | Price |  
|------|-------|  
| 200  | 100   |  

This is clearly a different input because **you are swapping the meaning of features** (price becomes size, and size becomes price).

---

### **What’s Different in Text Data?**  
Text is **not fixed like numerical features**. The order of words carries meaning in a sentence. For example:  

- **"I like pizza"** → Positive statement.  
- **"Pizza likes me"** → Doesn't make sense the same way.  

If you treat the sentence as a "bag of words" (like ANN does), both sentences look identical because the words are the same, just jumbled. ANN can't understand the difference.

---

### **How RNN Solves This Issue**  
RNN **reads the text in order, word by word**, and keeps "memory" of the previous words.  

For example, if RNN processes:  
1. **"I"** → Learns the subject (who is doing the action).  
2. **"like"** → Adds meaning (what the subject feels).  
3. **"pizza"** → Completes the idea (what they feel about).  

So, RNN understands the sentence's meaning because it processes the sequence.

---

### **Why ANN Can't Do This?**  
ANN looks at all words at once, like dumping all ingredients into a pot without caring about their order. This works for fixed numerical features but **fails for sequential data like text**, where the order changes the meaning.

---

### **Summary**  
- In **ANN**, order matters because you define the meaning of positions (e.g., price vs. size).  
- In **RNN**, order matters because it processes inputs one by one and captures the meaning of sequences, which ANN cannot do.  

---
![image](https://github.com/user-attachments/assets/dd1f1db4-3463-4059-b6a8-8190389bc952)


### **Why is the Order Important in ANN?**  

1. **Fixed Mapping to Features**:  
   Each input position corresponds to a specific feature in the dataset.  
   - Example: For a house prediction model:  
     - Position 1 = Price  
     - Position 2 = Size  
   If you reverse the order, the meaning of the input changes, and the model learns incorrect relationships.

2. **Model Can't Guess the Meaning**:  
   ANN doesn’t understand the "meaning" of features. It just associates numbers at specific positions with their weights. Swapping the order will confuse it.

---

### **How is this Different from RNN?**  

1. **Sequential Understanding**:  
   RNN processes inputs one step at a time, in a sequence. For example, when processing text or time-series data, RNN learns relationships between consecutive steps (like one word after another).  

   - In ANN, there’s no such notion of “time steps” or “sequence.” It just takes all inputs simultaneously.

2. **Focus on Context**:  
   RNN maintains context from earlier steps (e.g., previous words in a sentence) to understand the current input. ANN lacks this "memory."

---

### **Key Takeaway**  
- **In ANN**, order matters **only for feature positions** because they have fixed meanings.  
- **In RNN**, order matters for **sequence processing**, like in text, where the context depends on previous inputs.

Both have **order dependency**, but **how and why** it matters is what sets them apart.

---

### **1. The Problem Statement: Sentiment Analysis**  
We are trying to classify sentences into categories like positive (1) or negative (0). For example:  
- **"The food is good"** → Positive (1)  
- **"The food is bad"** → Negative (0)  
- **"The food is not good"** → Negative (0)  

---

### **2. Text Preprocessing and Feature Representation**  
Before we can feed text data into any neural network, we need to convert it into numbers (vectors) because neural networks can’t directly process text.  
#### Common Techniques:  
1. **Bag of Words (BoW)**:  
   - Each unique word becomes a feature (column in a table).  
   - Sentences are represented as vectors based on word presence (1 if the word exists, 0 otherwise).  

   **Example**:  
   Vocabulary = [food, good, bad, not]  
   - "The food is good" → [1, 1, 0, 0]  
   - "The food is bad" → [1, 0, 1, 0]  
   - "The food is not good" → [1, 1, 0, 1]  

2. **TF-IDF**: Assigns importance to words based on how often they appear in the sentence and how unique they are across all sentences.  

3. **Word Embeddings (e.g., Word2Vec)**: Converts each word into a fixed-size vector (e.g., 300 dimensions) that captures its meaning.

---

### **3. Issues with ANN (Artificial Neural Network) for Sequential Data**  
When we use ANN to process text data, **two key problems arise**:  
1. **Loss of Sequence Information**:  
   - ANN treats the sentence as a flat structure. It doesn't consider the order of words.  
   - Example: **"The food is not good"** and **"The food is good not"** would look the same in Bag of Words representation.  

2. **All Words Processed at Once**:  
   - ANN takes the entire sentence (vector) as input at once, which doesn't respect the sequential nature of language.  

**Why is sequence important?**  
In sentences, the order of words matters. For example:  
- **"The food is not good"** means something very different from **"The food is good."**

---

### **4. Solution: Recurrent Neural Networks (RNNs)**  
RNNs are specifically designed to handle sequential data by processing one word at a time and maintaining "memory" of previous words.  

#### How RNNs Work:  
- Each word is processed step-by-step (one word at a time).  
- The network maintains a hidden state (memory) that captures information about the sequence so far.  
- Example: For the sentence **"The food is good"**, RNN processes it like this:  
  1. **Word 1 ("The")** → Updates hidden state.  
  2. **Word 2 ("food")** → Updates hidden state based on "The" and "food."  
  3. **Word 3 ("is")** → Updates hidden state based on "The," "food," and "is."  
  4. **Word 4 ("good")** → Final output based on the entire sequence.  

#### Why RNNs Are Better for Sequential Data:  
1. They preserve the sequence of words.  
2. They update their memory at each step, maintaining context.  

---

### **5. ANN vs. RNN: Key Difference**  
| Feature                 | ANN                         | RNN                           |
|-------------------------|-----------------------------|-------------------------------|
| Input Handling          | Entire sentence at once     | Word by word (sequentially)   |
| Sequence Information    | Lost                        | Preserved                    |
| Best For                | Non-sequential data         | Sequential data (e.g., text, time series) |

---

### **Example to Compare**  
Let’s take a simple sentence: **"The food is good."**

- **Using ANN**:  
  Converts the sentence into a fixed-size vector ([1, 1, 0, 0]) and feeds it to the network. Sequence is lost.  

- **Using RNN**:  
  Processes words one at a time:  
  1. **"The" → Updates memory.**  
  2. **"food" → Combines "The" + "food."**  
  3. **"is" → Combines "The food" + "is."**  
  4. **"good" → Combines "The food is" + "good" → Final Output.**  

---

### Explanation of Forward Propagation in Simple RNN

Forward propagation in a Simple Recurrent Neural Network (RNN) involves processing sequential data step-by-step through time, capturing the context of previous inputs via feedback loops. Here's a simplified explanation based on the given transcript:

---

#### **Key Concepts**
1. **Input Data Representation**:
   - Input words (e.g., "The food is good") are converted into numerical vectors for processing.
   - In this case, **One-Hot Encoding** is used:
     - Each unique word in the vocabulary gets a binary vector with one `1` (representing its position) and the rest `0`.

2. **Architecture Overview**:
   - The RNN consists of:
     - **Input Layer**: Receives the encoded vectors.
     - **Hidden Layer**: Processes inputs and retains context through feedback.
     - **Output Layer**: Generates predictions (e.g., classification results).

3. **Feedback Mechanism**:
   - Outputs from the hidden layer at the current timestamp are passed to the next timestamp as additional context (feedback loop).
   - ![image](https://github.com/user-attachments/assets/b6826622-565c-449e-b907-6341aac9639c)


---

#### **Forward Propagation Steps**
1. **Input Layer**:
   - At timestamp ( t=1 ), the first word vector (e.g., `10000` for "The") is fed into the network.
   - The number of inputs matches the vector size (e.g., 5  for a vocabulary of 5 words).

2. **Hidden Layer**:
   - Each input vector interacts with the **hidden neurons** through weight matrices and biases:
     ![image](https://github.com/user-attachments/assets/cf1d3871-0d12-4897-ba6c-65a54a1c900e)

   - The hidden layer output retains information from current and previous words.

![image](https://github.com/user-attachments/assets/3b497a40-14ae-491f-abd7-5f028792d0c6)


---
![image](https://github.com/user-attachments/assets/71519c12-3522-4792-a79a-53649241d85a)

---
#### **Example**
- **Sentence**: "The food is good"
 ![image](https://github.com/user-attachments/assets/6796b805-b7e6-40e9-93f8-e63a47c91584)

---

#### **Trainable Parameters Calculation**
- **Inputs**: \( 5 \) (vocabulary size)
- **Hidden Neurons**: \( 3 \)
- **Weights and Biases**:
  ![image](https://github.com/user-attachments/assets/7c7b2b8a-2a44-4f6d-b0c0-11c32b97ed8b)


---

#### **Key Takeaway**
- RNNs maintain sequence context by retaining hidden states and feedback loops across timestamps.
- Forward propagation ensures each input word impacts future predictions, enabling sequential understanding.
---
