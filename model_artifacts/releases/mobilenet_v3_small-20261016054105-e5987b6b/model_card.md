# Model card: mobilenet_v3_small-20261016054105-e5987b6b

**Intended use:** triage aid that estimates whether an image resembles the AI-generated class
seen in training (Defactify). It is not an authenticity oracle or a fraud detector.

**Architecture:** mobilenet_v3_small (MobileNet_V3_Small_Weights.DEFAULT), input 128x128.
**Dataset:** Defactify / 2026-10-11; manifest sha256 `e5987b6bdc5cb059221e4faa0c4292640db070ae3a2f83195cac595e4625082b`.
**Score type:** calibrated probability (temperature scaling, fitted on validation). Decision threshold 0.303 with abstain margin 0.10.
**MLflow run:** 63471e0d7a0c449d84db380b8d6827a2. **Git commit:** f06f664ad71c77b21bdbbe201c74e2f06bef5a70.

## Measured results (from the run that produced this bundle)
- **validation** (n=6292): precision 0.9349, recall 0.9721, F1 0.9531, ROC-AUC 0.9467, PR-AUC 0.9879, Brier 0.0660
- **internal_test** (n=45000): precision 0.9448, recall 0.9378, F1 0.9413, ROC-AUC 0.9386, PR-AUC 0.9863, Brier 0.0876

## Out of scope / not demonstrated
- Detecting every kind of deepfake, face swap, or manipulated media.
- Proving that a document, offer or claim is genuine or fraudulent.
- Concluding an image is authentic because the score is low.
- Real-world performance: results on Defactify reflect a narrow benchmark domain.

## Known limitations
- Results on this dataset do not establish detection of deepfakes or real-world fakes.
- Check dataset licence/terms before redistributing any images.
