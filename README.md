# End-to-End Deep Learning Project using Simple RNN

This project performs **IMDB movie review sentiment analysis** using a **Simple RNN (Recurrent Neural Network)** model.  
Given a movie review, the model predicts whether the sentiment is **positive** or **negative**, along with the probability score.

---

## 📂 Project Structure


---

## 📜 Workflow

1. **Embedding Testing**  
   - In `embedding.ipynb`, we experimented with the **Embedding layer** on small examples to understand how it works before implementing it in the main model.

2. **Model Training**  
   - `Simple_RNN.ipynb` handles:
     - Data cleaning and preprocessing
     - Tokenization and padding of sequences
     - Defining and training the Simple RNN model
     - Saving the trained model as `Simple_RNN_imdb.h5`

3. **Prediction**  
   - `prediction.ipynb` loads the trained `.h5` model and predicts sentiments for test data.

4. **Deployment**  
   - `main.py` contains a **Streamlit** app to:
     - Accept a user’s movie review as input
     - Return predicted sentiment (**Positive** / **Negative**) with probability

---

## 🛠 Installation & Usage

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yokeswarans/End-to-end-deep-learning-project-using-the-simple-rnn.git
cd End-to-end-deep-learning-project-using-the-simple-rnn 
pip install -r requirements.txt
streamlit run main.py
