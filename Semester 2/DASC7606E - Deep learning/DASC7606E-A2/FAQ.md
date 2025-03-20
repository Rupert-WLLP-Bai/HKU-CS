# Frequently Asked Questions (FAQ)

Here are answers to some common questions about the assignment:

**Q1: What is the difference between `bert-base-cased` and `bert-base-uncased`?**

**A1:**

- **`bert-base-cased`:** This model distinguishes between uppercase and lowercase letters. For example, it treats "Apple" and "apple" as different tokens. This is generally preferred for NER because capitalization can be an important feature for identifying named entities (e.g., "Apple" the company vs. "apple" the fruit).
- **`bert-base-uncased`:** This model converts all text to lowercase before tokenization. It treats "Apple" and "apple" as the same token.

**For this assignment, we recommend using `bert-base-cased` (or a similar cased model) unless you have a specific reason to use an uncased model.**

**Q2: How do I handle tokens that are split into multiple subword tokens during tokenization?**

**A2:** This is a crucial aspect of the assignment. When a word is split into multiple subword tokens, you need to align the labels accordingly:

1. The **first subword token** gets the **original label** of the word.
1. The **subsequent subword tokens** tokens inside a word but not at the beginning, we replace the B- with I- (since the token does not begin the entity):

**Example:**

- Original word: `Apple`
- Label: `B-CORP`
- Tokenized: `['App', '##le']`
- Aligned labels: `['B-CORP', 'I-CORP']`

**Q3: What should the output dimensions of my model be?**

**A3:** Your model should output a tensor of shape `(batch_size, sequence_length, num_labels)`, where:

- `batch_size` is the number of sentences in a batch.
- `sequence_length` is the maximum number of tokens in a sentence (after tokenization).
- `num_labels` is the number of unique entity labels in your dataset **plus one for the 'O' (Outside) label**. In the MultiCoNER dataset, you have 6 entity types (PER, LOC, CORP, GRP, PROD, CW) plus 'O', so `num_labels` should be 13 (B-PER, I-PER, B-LOC, I-LOC, B-CORP, I-CORP, B-GRP, I-GRP, B-PROD, I-PROD, B-CW, I-CW, O).

**Q4: How do I use the `seqeval` library for evaluation?**

**A4:** The `seqeval` library is designed for evaluating sequence labeling tasks like NER. Here's a basic example:

```python
from seqeval.metrics import classification_report

# True labels (replace with your actual data)
y_true = [["B-CORP", "O", "B-CORP", "O", "O", "O", "B-CORP", "O", "O"]]
# Predicted labels (replace with your model's predictions)
y_pred = [["B-CORP", "O", "B-CORP", "O", "O", "O", "B-CORP", "O", "O"]]

print(classification_report(y_true, y_pred))
```

**Important:**

- Make sure your true and predicted labels are lists of lists, where each inner list represents the labels for a single sentence.
- Remove the special label (e.g., -100) from your predicted and true labels before passing them to `seqeval`.

**Q5: My model is training very slowly. What can I do?**

**A5:** Training transformer models can be time-consuming. Here are some tips to speed up training:

- **Use the HKU GPU Farm:** This is the most effective way to accelerate training.
- **Reduce Batch Size:** If you're running out of memory, try reducing the batch size.
- **Mixed Precision Training:** If your GPU supports it, consider using mixed precision training (e.g., with the `accelerate` library) to reduce memory usage and speed up computation.
- **Gradient Accumulation:** If you need a larger effective batch size but are limited by memory, you can use gradient accumulation to simulate a larger batch size by accumulating gradients over multiple smaller batches.
- **Optimize Data Loading:** Ensure your data loading pipeline is efficient.

**Q6: Can I use a different pre-trained transformer model than `bert-base-cased`?**

**A6:** Yes, you are encouraged to experiment with different pre-trained models (e.g., RoBERTa, ELECTRA, LLM, etc.). Just make sure to use the appropriate tokenizer and model class for the model you choose and make sure that you meet the time and resource constraints of the assignment.

**Q7: What should I do if my code has errors?**

**A7:**

1. **Read the error message carefully:** Error messages often provide valuable clues about the source of the problem.
1. **Use a debugger:** Debuggers allow you to step through your code line by line and inspect the values of variables.
1. **Print intermediate values:** Use `print()` statements to check the values of variables and ensure they are what you expect.
1. **Search online:** Many common errors have solutions available online (e.g., on Stack Overflow).
1. **Ask for help:** If you're still stuck, attend office hours or post your question on the course discussion forum.

**Q8: What are some common mistakes to avoid?**

**A8:**

- **Incorrect Label Alignment:** This is the most common source of errors. Double-check your label alignment logic.
- **Using the Wrong Tokenizer:** Make sure you are using the correct tokenizer for your chosen pre-trained model.
- **Not Handling the Special Label (-100) Properly:** Remember to ignore the special label during loss calculation and evaluation.
- **Overfitting:** If your model performs very well on the training set but poorly on the test set, it might be overfitting. Try using regularization techniques or reducing the model's complexity.
- **Not Reporting All Metrics:** Report F1-score, precision, and recall, and include a confusion matrix.
- **Insufficient Analysis:** Provide a thorough analysis of your results and suggest concrete ways to improve your model.

**Q10: Where can I find more information about NER and transformer models?**

**A10:**

- **Hugging Face Transformers Documentation:** [https://huggingface.co/transformers/](https://huggingface.co/transformers/)
- **The original BERT paper:** [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
- **Stanford CS224N: Natural Language Processing with Deep Learning:** [https://web.stanford.edu/class/cs224n/](https://web.stanford.edu/class/cs224n/)
- **Speech and Language Processing (3rd ed. draft) by Jurafsky & Martin:** [https://web.stanford.edu/~jurafsky/slp3/](https://web.stanford.edu/~jurafsky/slp3/) (Chapter on Sequence Labeling and NER)

If you have any other questions, please don't hesitate to ask during office hours or on the course discussion forum!
