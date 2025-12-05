# 🛡️ NeuroScan: Robustness Analysis of Edge AI

The Problem: Medical AI needs to be portable (for phones) but also secure against errors. The Solution: I built a Quantized MobileNetV2 (73% smaller than standard) and tested it against adversarial attacks.

Key Findings:

Accuracy: 94% on Test Data.

Efficiency: Compressed from 9MB to 2.5MB.

Security Flaw: A noise injection of just 0.05 caused the AI to misdiagnose a tumor as healthy.

Code Structure:

NeuroScan: Vulnerability Analysis of Edge-Deployed Medical AI.ipynb: The training and attack experiments.

app.py: A live demo of the lightweight model.
