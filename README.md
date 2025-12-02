# MU3D-Lie-detection
Multimodal Lie Detection (MU3D)

This project explores automatic lie detection using multimodal deep learning.
The goal is to predict whether a speaker is telling the truth or lying based on:

Text – what is said (transcriptions)

Audio – how it is said (tone, rhythm, intonation, pauses)

Video – non-verbal signals (facial expressions, micro-expressions, gestures)

Humans themselves struggle with this task. The idea is to leverage AI models (NLP + audio + vision) to capture subtle patterns that may correlate with deception.

1. Problem Definition

Task: Binary classification – predict truth vs lie from observed signals.

Modalities:

Text: transcripts of the spoken content

Audio: prosodic features (pitch, energy, tempo, pauses, etc.)

Video (future work): facial expressions, head movements, micro-expressions

The current repository focuses primarily on text and audio, with a roadmap to extend to video.

2. Existing Approaches (High-Level Overview)
2.1 Unimodal Approaches

Each modality is used separately.

Text

Classical pipeline

Preprocessing + TF-IDF

Classifiers: Logistic Regression, SVM, Random Forest

Modern pipeline

Pretrained Transformers (e.g. BERT, RoBERTa)

Fine-tuning for binary classification (truth vs lie)

Audio

Features

MFCCs (Mel-Frequency Cepstral Coefficients)

Pitch (fundamental frequency)

Energy, jitter, shimmer, etc.

Models

RNNs / LSTMs, CNNs

Speech-oriented models such as wav2vec 2.0

Video

Features

Facial action units (AUs)

Facial landmarks / head pose

Models

2D / 3D CNNs

Video transformers (e.g. TimeSformer, SlowFast)

2.2 Multimodal Approaches

Combine several modalities: text, audio, video.

Early Fusion

Extract embeddings for each modality (text vector, audio vector, video vector)

Concatenate them and feed into a joint network.

Pros:

Learns interactions between modalities directly.

Cons:

Sensitive to missing modalities

Requires careful temporal alignment.

Late Fusion

Train a separate model per modality

Combine outputs (e.g. average probabilities, majority vote).

Pros:

Robust to missing modalities.

Each branch can be tuned independently.

Cons:

Loses fine-grained interactions between modalities.

Hybrid / Attention-Based Fusion

Use attention (self-attention or cross-attention) to let modalities influence each other.

Example: audio frames become more important when facial expressions show tension.

Pros:

Better alignment and exploitation of complementary signals.

Cons:

More complex and expensive to train.

Examples of multimodal models:

MMBT (Multimodal Bitransformer)

MulT (Multimodal Transformer)

MISA

DEVA (progressive fusion with attention blocks)

3. Evaluation Metrics

To assess the quality of the models, we use:

Accuracy – % of correct predictions

Simple to interpret, but misleading under class imbalance.

Precision (for “lie”) –

Among the samples predicted as lie, how many are actually lies?

Recall (for “lie”) –

Among all real lies, how many did the model detect?

F1-score – harmonic mean of precision and recall

Key metric when we care about both false positives and false negatives.

ROC-AUC – area under the ROC curve

Threshold-independent view of the trade-off between TPR and FPR.

Balanced Accuracy – average recall per class

Useful when classes are imbalanced.

In more advanced benchmarks, we may also report:

Macro F1 – average F1 over all classes (treating them equally).

4. Datasets
MU3D (Chosen Dataset)
Property	Description
Modalities	Video, Audio, Text
Size	320 clips (80 participants × 4 clips)
Language	English
Labels	truth vs lie

Strengths

Balanced labels (truth / lie)

High-quality transcriptions

Well-documented and ethically collected

Limitations

Relatively small dataset

Single social/relational context

Mainly American population

Other Related Datasets (Not used yet)

MDPE

Modalities: video, audio, pre-extracted features (AUs, pose, etc.)

Language: Chinese

Pros: rich multimodal signals

Cons: no directly usable text transcripts in English

CMU-MOSEI

Modalities: video, audio, text

~23k clips in English

Pros: large scale, diverse emotions, well preprocessed

Cons: primary labels are affective (sentiment/emotion), not truthfulness

5. MU3D: Features Used

The MU3D codebook contains:

VideoID – unique id for each clip

Valence – 0 = negative, 1 = positive

Veracity – target label:

0 = lie

1 = truth

Sex, Race – demographic attributes of the speaker

VidLength_ms / VidLength_sec – duration of the video

WordCount – number of words in the transcript

Accuracy – proportion of annotators who correctly guessed lie vs truth

TruthProp – proportion of annotators who believed the clip was “true”

Attractive – perceived attractiveness (1 to 7)

Trustworthy – perceived trustworthiness (1 to 7)

Anxious – perceived anxiety (1 to 7)

Transcription – full text of what is said

The current pipeline mainly uses:

Transcription (text)

Veracity (labels)

audio_path (for feature extraction)

A subset of additional features as needed.

6. Project Roadmap / Pipeline

The repository is structured around a progressive construction of lie detection models:

Step 1 – Simple Text-Only Prototype

Input: transcriptions only

Preprocessing + TF-IDF vectorization

Classifiers: Logistic Regression or Random Forest

Baseline metrics: accuracy, precision, recall, F1

Step 2 – Advanced Text-Only Model

Encode text using BERT (or DistilBERT)

Two options:

Full fine-tuning (end-to-end)

Frozen embeddings + dense classifier on top

Techniques:

dropout for regularization

class_weight='balanced' for imbalanced classes

Cross-validation for robust evaluation

Step 3 – Add Audio Modality

Extract audio features:

MFCCs, pitch, jitter, etc. (first prototype)

Then embeddings from wav2vec 2.0 (final multimodal model)

Normalize features

Models:

Simple MLP, CNN or LSTM (depending on feature type)

Fusion with text:

Early fusion – concatenate text embedding (BERT) and audio embedding into a joint classifier.

Step 4 – Add Video Modality (Future Work)

Extract facial landmarks / AUs using OpenFace or similar tools

Optionally:

3D CNN or ResNet on cropped faces

Three-way fusion: text + audio + video

Step 5 – Adaptive Fusion (Inspired by DEVA)

Use cross-modal attention to let modalities influence each other

Learn dynamic weights per modality (context-dependent)

Incorporate temporal modeling with BiLSTM or multimodal Transformers

Step 6 – Optimization & Robust Evaluation

Hyperparameter tuning (GridSearch / Optuna)

Regularization and potential oversampling (e.g. SMOTE)

Data augmentation:

Visual: random crops, flips, brightness changes

Audio: noise, pitch/tempo shifts

Metrics:

Macro F1, AUC, balanced accuracy, per-class scores

7. Possible Extensions for the README

In the future, the README can be enriched with:

Architecture diagrams for:

Early / late / hybrid fusion

Current multimodal model (Text + Audio + [Video])

Example predictions:

Input snippet (text + spectrogram frame) → model output + explanation

Links to related open-source models and papers
