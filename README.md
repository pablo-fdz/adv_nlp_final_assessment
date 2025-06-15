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

## 3. Results
