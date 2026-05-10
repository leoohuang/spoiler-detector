# Multi-Source Spoiler Detector

**Student:** Tianle Huang  
**GitHub repository:** https://github.com/leoohuang/spoiler-detector  
**Hugging Face dataset:** https://huggingface.co/datasets/leoole/spoiler-detector  
**Working demo:** https://huggingface.co/spaces/leoole/spoiler-detector  

## Abstract

This project implements a three-class movie spoiler detector for English reviews. The system classifies a review as `Safe`, `Mild Spoiler`, or `Major Spoiler`, instead of only using a binary spoiler/non-spoiler label. The final solution includes a custom processed dataset, several trained classifiers, an evaluation on a held-out test set, a GitHub repository, a Hugging Face dataset, and a working Hugging Face web demo. The best model was an RBF-kernel Support Vector Machine using sentence-transformer embeddings, with 0.575 accuracy and 0.572 macro F1 on the test set.

## Dataset Creation

The original plan was to use Reddit as an additional data source, but this was changed because Reddit API registration and approval can be slow and inconvenient for a small course project. Instead, the final dataset was created from two sources: IMDb movie reviews and GPT-generated synthetic review snippets.

IMDb was used as the main real-world source because it contains movie-review text and a binary spoiler indicator. However, the project required three labels rather than only spoiler/non-spoiler, so additional processing was needed. Non-spoiler IMDb rows were labeled as `Safe`. IMDb spoiler rows were further labeled as either `Mild Spoiler` or `Major Spoiler` using GPT-assisted severity labeling. The distinction was based on whether the text revealed general plot setup or non-critical information (`Mild`) versus revealing a key twist, death, ending, identity, solution, or final outcome (`Major`).

GPT-generated synthetic examples were added as a second source to improve coverage across the three target classes. The prompts asked for short English movie-review snippets and avoided obvious phrases such as "no spoilers", since such phrases could make the classification task artificially easy.

After cleaning, normalization, length filtering, deduplication, and labeling, the final dataset contained 4,773 examples:

| Source | Count |
|---|---:|
| IMDb | 4,000 |
| GPT synthetic | 773 |

| Label | Count |
|---|---:|
| Safe | 2,270 |
| Mild Spoiler | 1,616 |
| Major Spoiler | 887 |

To check labeling quality, I manually reviewed a sample of 100 Mild/Major examples. The manual labels had 93% exact agreement with the GPT-assisted severity labels. This suggests that the labels were reasonably consistent, although spoiler severity is still partly subjective.

## Method

The dataset was split into train, validation, and test sets using stratification by severity label. The final split contained 3,818 training examples, 477 validation examples, and 478 test examples.

For feature extraction, I used `sentence-transformers/all-mpnet-base-v2` to convert each review into a dense semantic embedding. These embeddings were then used to train and compare four classifiers:

| Classifier |
|---|
| Logistic Regression |
| Support Vector Machine with RBF kernel |
| Random Forest |
| Multi-Layer Perceptron |

This approach allowed the project to use strong pretrained language representations while keeping the classification models simple and easy to compare.

## Results

The best model on the held-out test set was the RBF-kernel SVM. The full comparison is shown below:

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| SVM RBF | 0.575 | 0.572 | 0.575 |
| Logistic Regression | 0.567 | 0.571 | 0.566 |
| MLP | 0.569 | 0.564 | 0.567 |
| Random Forest | 0.531 | 0.417 | 0.443 |

The best model performed most strongly on the `Safe` category, with an F1-score of 0.622. It reached 0.598 F1 for `Major Spoiler` and 0.497 F1 for `Mild Spoiler`. The `Mild Spoiler` class was the hardest category, which makes sense because the boundary between mild plot information and a major spoiler can be ambiguous even for human annotators.

Overall, the results show that three-level spoiler detection is feasible, but more difficult than binary spoiler detection. The moderate performance also reflects the difficulty of the task, the limited dataset size, and the subjective nature of spoiler severity.

## Deployment

The final demo was deployed as a Hugging Face Space using Gradio. A user can enter a movie review and receive a predicted class, confidence score, and probability distribution across the three labels. The model artifact is hosted in a separate Hugging Face model repository, while the dataset is hosted as a Hugging Face dataset. Large model and data files are not stored directly in the GitHub repository, which keeps the repository easier to inspect and clone.

## Reflection on Using AI

AI tools were useful in several parts of the assignment. I used GPT to help generate synthetic review examples, assist with severity labeling, write and revise code, and draft parts of the report. This made the project faster to develop, especially when designing the data pipeline and fixing small implementation issues.

At the same time, I had to check the AI output carefully. The most important risk was hallucination or overconfident labeling, especially when deciding whether a spoiler was mild or major. To reduce this risk, I manually inspected sampled labels, checked dataset distributions, compared multiple classifiers, and verified that the web demo actually worked on Hugging Face. I also reviewed the generated code and README instead of assuming that AI-produced text or code was automatically correct.

Using AI was most helpful as a programming and writing assistant, but the final responsibility still remained with me. I had to make the project decisions, verify the results, and ensure that the submitted repository, dataset, and demo matched the assignment requirements.

## Conclusion

This project satisfies the assignment requirements by creating a custom text classification dataset, training and evaluating several classifiers, and deploying a working Hugging Face demo. The project also demonstrates a practical workflow for adapting a data plan when an API source is difficult to use. The final system is not a production-level spoiler moderation tool, but it is a complete and reproducible course-project prototype for multi-class spoiler severity detection.

## References

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *EMNLP-IJCNLP 2019*.

Hugging Face. (n.d.). Spaces and datasets documentation. https://huggingface.co/docs

IMDb review data used in this project was processed for course-project text classification purposes.

