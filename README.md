# Fatigue Predictor — Behavioral Markers of Cognitive Fatigue

A classical machine learning pipeline predicting cognitive fatigue from
behavioral performance markers, reaction time, error rate, and session
duration rather than raw physiological signal processing (e.g. EEG),
as an honestly-scoped project at my current technical level.

## Motivation
Cognitive fatigue is a well-established phenomenon in psychology, with
real consequences in clinical, educational, and safety-critical settings.
This project explores whether fatigue can be predicted purely from
behavioral performance patterns.

## What this does
- Loads trial-level performance data (reaction time, error rate, session length)
- Labels each trial as fatigued/not fatigued based on a threshold rule
- Trains a logistic regression classifier to predict fatigue from the three features
- Evaluates accuracy and confusion matrix on held-out test data
- Visualizes the separation between fatigued and non-fatigued trials

## Status and honest limitations
Currently built and tested on synthetic mock data with a clean,
rule-based threshold hence the current 100% test accuracy, which
reflects the simplicity of the synthetic labels rather than real-world
predictive power. Real behavioral data would very likely show more
ambiguity and lower accuracy. Next step: testing against a real public
fatigue/reaction-time dataset.

## Tools
Python, pandas, scikit-learn (LogisticRegression), matplotlib
