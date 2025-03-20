# HKU-DASC7606-A2: Named Entity Recognition (NER) with Transformer Models

**Course:** HKU DASC-7606 (2024-2025)
**Assignment:** 2 - Named Entity Recognition (NER) with Transformer Models

**Important:** This codebase is exclusively for HKU DASC 7606 (2024-2025). Please do not upload your solutions or this codebase to any public platforms. All rights reserved.

## 1. Introduction: Unveiling Entities in Text

### 1.1. What is Named Entity Recognition (NER)?

Imagine you're building a system to automatically extract key information from news articles. You'd want to identify names of people, organizations, locations, dates, and other important entities. This is precisely what **Named Entity Recognition (NER)** does. It's a fundamental task in Natural Language Processing (NLP) that involves automatically identifying and classifying named entities in text into predefined categories.

**Why is NER Important?**

NER is a cornerstone of many NLP applications, including:

* **Information Extraction:**  Pulling structured data (e.g., who did what, where, and when) from unstructured text.
* **Question Answering:** Helping systems understand the entities involved in a question to provide accurate answers.
* **Knowledge Graph Construction:** Building interconnected networks of entities and their relationships.
* **Machine Translation:** Ensuring that named entities are translated correctly.
* **Text Summarization:** Identifying key entities to include in summaries.

### 1.2. Transformers: A Revolution in NER

Traditional NER systems often relied on handcrafted features and complex rule-based systems. However, the advent of **transformer-based models** like BERT, RoBERTa, and others has revolutionized the field. These models excel at understanding the context of words, leading to significant improvements in NER accuracy.

**How do Transformers Help?**

* **Contextual Embeddings:** Transformers generate word representations that capture the meaning of a word based on its surrounding words. This is crucial for NER, as the same word can be different entity types depending on the context (e.g., "Apple" the company vs. "apple" the fruit).
* **Pre-training and Fine-tuning:** Transformers are typically pre-trained on massive amounts of text data, learning general language understanding. This pre-trained knowledge can then be fine-tuned for specific tasks like NER with relatively smaller datasets.

### 1.3. Your Mission in This Assignment

In this assignment, you will dive into the world of NER using transformer models. You will:

* **Master the Basics:** Gain a solid understanding of how transformer models work and how they are applied to NER.
* **Get Hands-On:** Learn to preprocess data for NER, including tokenization and label alignment, which are crucial steps for successful model training.
* **Build and Train:** Fine-tune a pre-trained transformer model on a subset of the well-known MultiCoNER NER dataset.
* **Evaluate and Analyze:** Assess your model's performance using standard metrics (F1-score, precision, recall) and analyze its strengths and weaknesses.
* **Become an NER Expert:** Develop a deeper understanding of the challenges and potential improvements in NER systems.

## 2. Setting Up Your NER Lab

### 2.1. HKU GPU Farm: Your High-Performance Playground (Recommended)

The HKU GPU Farm provides the computational power you need for efficient training of transformer models. Follow the provided [quickstart guide](https://www.cs.hku.hk/gpu-farm/quickstart) to set up your environment. This is the recommended approach for this assignment.

### 2.2. Local Setup: For the Resourceful

If you have a powerful local machine with a suitable GPU and have experience setting up deep learning environments, you can work locally. Ensure you have the necessary software (CUDA, cuDNN) installed and configured correctly.

### 2.3. Environment Setup: Your Toolkit

**Python:** This code is tested with Python 3.11.10.

**Virtual Environment (Recommended):**  Use Anaconda to manage your project's dependencies and avoid conflicts:

```bash
conda create -n nlp_env python=3.11.10
conda activate nlp_env
```

**Install Packages:**

```bash
pip install -r requirements.txt
```

## 3. Embarking on Your NER Journey

### 3.1. Dataset: The MultiCoNER NER Dataset

You will be working with a subset of the famous **MultiCoNER NER dataset**. This dataset contains sentences annotated with six types of named entities:

* **PER:** Person, i.e. names of people
* **LOC:** Location, i.e. locations/physical facilities
* **CORP:** Corporation, i.e. corporations/businesses
* **GRP:** Group, i.e. all other groups
* **PROD:** Product, i.e. consumer products
* **CW:** Creative Work, i.e. movies/songs/book titles

The dataset is provided in the **BIO format**:

* **B-TYPE:** Beginning of an entity of type TYPE
* **I-TYPE:** Inside an entity of type TYPE
* **O:** Outside any entity

**Example:**

| Token  | Label  |
| :----- | :----- |
| John   | B-PER  |
| Doe    | I-PER  |
| works  | O      |
| at     | O      |
| Google | B-CORP |
| in     | O      |
| London | B-LOC  |
| .      | O      |

**Data Format:** The dataset is in JSONL format (one sentence per line with token-label pairs).

### 3.2. Preprocessing: Preparing Your Data for the Transformer

Preprocessing is a critical step to ensure your data is in the correct format for the transformer model.

**Steps:**

1. **Load the Dataset:** Load **MultiCoNER NER dataset** into a suitable data structure.
2. **Tokenization:** Use a tokenizer from the Hugging Face `transformers` library (e.g., `BertTokenizerFast` for `bert-base-cased`) to split sentences into tokens.
    * **Hint:** Transformers often use subword tokenization (e.g., "playing" might become "play" and "##ing").
3. **Label Alignment:** This is the most crucial part! Since subword tokenization can split a single word into multiple tokens, you need to align the labels accordingly.
    * **Rule:** If a word is split into multiple subword tokens, the first subword token gets the original label, and the subsequent subword tokens get a special label (e.g., -100, which will be ignored during loss calculation).
    * **Example:**
        * Original: `John works at Apple.`
        * Labels: `['B-PER', 'O', 'O', 'B-CORP', 'O']`
        * Tokenized: `['John', 'work', '##s', 'at', 'App', '##le', '.']`
        * Aligned Labels: `['B-PER', 'O', 'O', 'O', 'B-CORP', 'I-CORP', 'O']`

### 3.3. Task Description: Your NER Adventure

Your mission is divided into three key tasks:

1. **Preprocessing and Tokenization (datasets.py):** Implement the data loading, tokenization, and label alignment logic.
2. **Model Implementation (model.py):** Define the transformer-based NER model using the Hugging Face `transformers` library and/or `torch`.
3. **Evaluation (evaluation.py):** Implement the code to calculate F1-score, precision, and recall for your model's predictions.

### 3.4. Code Structure: Your Project's Blueprint

```text
project/
|-- test/            # Test cases for evaluation
├── constants.py     # Constants and configurations
├── model.py         # Transformer model for NER
├── dataset.py       # Data loading, preprocessing, and tokenization
├── trainer.py       # Training loop and logic
├── tokenizer.py     # Tokenizer for the transformer model
├── evaluation.py    # Evaluation metrics (F1, precision, recall)
├── utils.py         # Utility functions
├── main.py          # Main script to run training and evaluation
├── requirements.txt # Python dependencies
├── FAQ.md           # Frequently asked questions
└── README.md        # Your project's documentation
```

**Running the Code:**

```bash
python main.py
```

### 3.5. Assignment Tasks: Your Path to NER Mastery

#### Task 1: Preprocessing and Tokenization (datasets.py)

* **Objective:**  Correctly load, tokenize, and align labels for the MultiCoNER dataset.
* **File:** `dataset.py`
* **Instructions:** Complete all sections marked with "`Write Your Code Here`".
* **Hints:**
  * Use the `datasets` library to load the dataset.
  * Use the appropriate tokenizer from the `transformers` library.
  * Carefully implement the label alignment logic, paying attention to subword tokens.
  * Consider adding a function to visualize a few examples of tokenized sentences with aligned labels to verify your implementation.
  * You can use any other dataset for training and evaluation if you prefer, but keep it in mind that: you need to make sure that the `test` split of your dataset is in the same as the `test` split of MultiCoNER dataset.
  * After you finished this task, please make sure that you pass the test case in `test/test_align_labels_with_tokens.py` by running the following command:

    ```bash
    pytest test/test_dataset.py
    ```

    If you pass the test case, you can move on to the next task.

#### Task 2: Model Implementation (model.py)

* **Objective:** Define a transformer-based model for NER.
* **File:** `model.py`
* **Instructions:** Complete the sections marked with "`Write Your Code Here`".
* **Hints:**
  * Use a pre-trained transformer model from the `transformers` library (e.g., `BertForTokenClassification` for `bert-base-cased`).
  * Ensure the model's output layer has the correct number of output units (equal to the number of entity labels).

#### Task 3: Evaluation (evaluation.py)

* **Objective:** Calculate F1-score, precision, and recall to evaluate your model's performance.
* **File:** `evaluation.py`
* **Instructions:** Complete the sections marked with "`Write Your Code Here`".
* **Hints:**
  * Use the `seqeval` library for computing these metrics for sequence labeling tasks.
  * Remember to handle the special label (e.g., -100) used during label alignment.
  * After you finished this task, please make sure that you pass the test case in `test/test_evaluation.py` by running the following command:

    ```bash
    pytest test/test_evaluation.py
    ```

    If you pass the test case, you can move on to the next task.


### 3.6. Submission: Packaging Your NER Masterpiece

**If your student ID is 30300xxxxx, organize your submission as follows:**

```text
30300xxxxx.zip
|-- test/            # Test cases for evaluation
├── constants.py     # Constants and configurations
├── model.py         # Transformer model for NER
├── dataset.py       # Data loading, preprocessing, and tokenization
├── trainer.py       # Training loop and logic
├── tokenizer.py     # Tokenizer for the transformer model
├── evaluation.py    # Evaluation metrics (F1, precision, recall)
├── utils.py         # Utility functions
├── requirements.txt # Python dependencies
└── main.py          # Main script to run training and evaluation
```

* **Code Files:** All your modified code files.
* **Submission Format:** Zip archive with your student ID as the filename.

### 3.7. Submission Deadline

**Deadline:** April 14th (23:59 GMT +8)

**Late Submission Policy:**

* 10% penalty within 1 day late.
* 20% penalty within 2 days late.
* 50% penalty within 7 days late.
* 100% penalty after 7 days late.

## 4. Grading: Your Path to Recognition

Your submission will be evaluated based on criterion:

### 4.1. Model Performance

We will re-run your `main.py` script to evaluate your model's performance on the test set.

**Important Considerations:**

1. **Error-Free Execution:** Your code must run without any errors.
2. **Correct Training and Evaluation:** Ensure your model is trained and evaluated correctly according to the instructions.
3. **Accurate Metrics:** The evaluation script must compute F1-score, precision, and recall accurately.
4. **Reasonable Performance:** Your model should achieve a reasonable F1-score on the test set.
5. **Execution Time:** The execution time should be less than 12 hours.

**Grading Breakdown (based on F1-score on the test set):**

* **F1-score >= 0.85:** Full marks (100%)
* **F1-score >= 0.80:** 90% of the marks
* **F1-score >= 0.75:** 80% of the marks
* **F1-score >= 0.70:** 70% of the marks
* **F1-score >= 0.65:** 60% of the marks
* **F1-score >= 0.60:** 50% of the marks
* **F1-score < 0.60/Fail to reproduce/Overtime:** No marks (0%)

## 5. Going Beyond: Extensions for the Ambitious (Optional)

If you're eager to explore further, here are some optional extensions:

* **Experiment with Different Transformer Models:** Try different pre-trained models (e.g., RoBERTa, ELECTRA) and compare their performance.
* **Hyperparameter Tuning:** Use techniques like grid search or Bayesian optimization to find the optimal hyperparameters for your model.
* **Error Analysis:** Implement more sophisticated error analysis techniques to gain deeper insights into your model's weaknesses.
* **Data Augmentation:** Explore data augmentation techniques to increase the size of your training data and potentially improve performance.
* **Incorporate External Knowledge:** Investigate how to incorporate external knowledge sources (e.g., knowledge bases) to enhance your NER system.
* **Try a Different Dataset:** Apply your code to a different NER dataset with different entity types.

## 6.  Need Help?

Please check [frequency asked questions](FAQ.md) (FAQ) first. If you have any questions or need clarification, feel free to reach out to the course instructor or teaching assistants.
