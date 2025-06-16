# Text Classification Task: Prediction of the Outcome of Swiss Federal Supreme Court cases

Source of the data: https://huggingface.co/datasets/rcds/swiss_judgment_prediction

## 1. Objective

### Task

The goal of this project is to explore the task of legal judgment prediction using various NLP techniques, like lexicon-based and transformer-powered classification. This task was a priori a challenging one, because deciding on a legal case usually requires an understanding of case complexities, legal precedence and national/international law.

### Dataset

- Our dataset comes from a multilingual (German, French, and Italian), diachronic (2000-2020) corpus of 85K cases from the Federal Supreme Court of Switzerland (FSCS). The FSCS is the last level of appeal in Switzerland and hears only the most controversial cases which could not have been sufficiently well solved by (up to two) lower courts. In its decisions, the FSCS often focuses only on small parts of previous decisions, discussing possible wrong reasoning by the lower courts. This makes these cases particularly challenging.
- The dataset is highly imbalanced containing more than 75% dismissed cases (85% in the "test" set). The label skewness makes the classification task quite hard and beating dummy baselines, e.g., predicting always the majority class, on microaveraged measures (e.g., Micro-F1) is challenging.

### Labels

This is a binary classification task, where a legal case can be either Dismissed (0) or Approved (1). When the FSCS "approves" a case, it means the Court has ruled in favor of the appellant — the party challenging the lower court’s decision. In practical terms, if the appeal is approved (accepted):
- The Supreme Court determines that the lower court made a legal error or violated constitutional rights.
- The contested decision is overturned (quashed) or sent back (remanded) to the lower court for a new decision in line with the Supreme Court’s findings.
- The appellant (the party who brought the appeal) wins the case at this stage.

If the appeal is rejected (dismissed):
- The Supreme Court upholds the lower court’s decision.
- The appellant’s challenge fails, and the original judgment stands.

In summary:

When the Swiss Federal Supreme Court approves a case, it means the appeal is successful and the lower court’s decision is overturned or modified in favor of the appellant. If the case is rejected, the lower court’s decision remains in force.

## 2. Main Findings

pablo-fdz: Can you create a brief summary for the main findings and the results, considering all of the insights highlighted throughout the notebooks?

GitHub Copilot: # Text Classification Task: Prediction of the Outcome of Swiss Federal Supreme Court cases

## 2. Main Findings

Our exploration of legal judgment prediction using the Swiss Federal Supreme Court dataset revealed several key insights:

### Dataset Challenges
- **Severe class imbalance**: With ~75% dismissed cases in the training set and ~85% in the test set, models naturally bias toward predicting dismissals.
- **Domain complexity**: Legal text classification requires specialized domain knowledge and language understanding.
- **Multilingual context**: The Swiss dataset's multiple languages (German, French, Italian) adds complexity, though we focused primarily on French documents.

### Model Performance
- **Domain-specific pre-training matters**: Models pre-trained on legal text (JuriBERT) consistently outperformed general language models on this specialized task.
- **Underfitting vs. overfitting**: With limited labeled data, the primary challenge was underfitting rather than overfitting - even with only 32 labeled examples, data augmentation helped more than regularization.
- **Zero-shot limitations**: Even advanced LLMs struggled with nuanced legal classification in zero-shot settings, defaulting to predicting the majority class.

### Efficiency Techniques
- **Data augmentation effectiveness**: Simple techniques like random word insertion and swap proved more effective than complex approaches like back-translation.
- **Knowledge distillation viability**: Student models achieved comparable performance to their teachers with significantly reduced size and faster inference.
- **Quantization trade-offs**: Converting model parameters to 8-bit integers substantially reduced model size with minimal performance impact.

## 3. Results

### Baseline Performance
- **Random classifier**: 63% accuracy (reflecting class distribution)
- **Rule-based classifier**: 62% accuracy using frequency of legal terms by area

### Limited Data Approaches (32 labeled examples)
- **BERT with standard fine-tuning**: ~60% AUC
- **JuriBERT with SetFit**: ~62% AUC (best performing approach with limited data)

### Scaling with More Data
- **Incremental training** showed initial improvement from 1% to 25% of training data, but performance gains plateaued afterward
- **Class imbalance** remained challenging even with full dataset

### Model Compression
- **Teacher model** (full CamemBERT): Highest accuracy but largest size
- **Distilled student model**: ~5% relative drop in accuracy with ~40% size reduction
- **Quantized model**: ~8% relative drop in accuracy with ~75% size reduction

### LLM-based Approaches
- **Zero-shot classification**: Good accuracy (87%) but only because of defaulting to majority class
- **Synthetic data generation**: Generated balanced datasets but introduced artificial patterns that sometimes reduced performance

Overall, the most promising approach combined domain-specific models (JuriBERT) with contrastive learning techniques (SetFit) and mild data augmentation, addressing both the domain complexity and data scarcity challenges inherent in legal text classification.