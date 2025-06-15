# Text Classification Task: Prediction of the Outcome of Swiss Federal Supreme Court cases

Source of the data: https://huggingface.co/datasets/rcds/swiss_judgment_prediction

## 1. Objective

# Just random quotes from the paper (for now)

- The FSCS is the last level of appeal in Switzerland and hears only the most controversial cases which could not have been sufficiently well solved by (up to two) lower courts. In their decisions, they often focus only on small parts of previous decision, where they discuss possible wrong reasoning by the lower court. This makes these cases particularly challenging.
- The dataset is highly imbalanced containing more than 75% dismissed cases. The label skewness makes the classification task quite hard and beating dummy baselines, e.g., predicting always the majority class, on microaveraged measures (e.g., Micro-F1) is challenging.


When the Federal Supreme Court of Switzerland "approves" a case, it means the Court has found in favor of the appellant—the party challenging the lower court’s decision. In practical terms, if the appeal is approved (accepted):
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