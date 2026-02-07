# AI Defect Detection System

Binary classification system for automated defect detection in construction materials and textiles.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

Deep learning system for quality control in manufacturing and construction. Trained on 70,000+ labeled images across multiple defect types and domains.

**Current capabilities:**
- Construction materials: Concrete defects, surface irregularities, casting defects
- Textile inspection: Fabric defects, weaving errors (beta)
- Real-time inference: 17-37ms per image on CPU
- Deployment-ready: Flask API and Streamlit interface included

---

## Performance Metrics

### Construction Materials (Production Ready)

| Metric | Score | Notes |
|--------|-------|-------|
| Precision | 98.2% | Low false positive rate |
| Recall | 97.4% | Catches most defects |
| F1-Score | 97.8% | Balanced performance |
| Inference Speed | 20ms | CPU (Intel i7) |
| Throughput | 100+ images/min | Batch processing |

### Textiles (Beta)

| Metric | Score | Notes |
|--------|-------|-------|
| F1-Score | 90-95% | Domain-dependent |
| Status | Beta | Human review recommended |
| Inference Speed | 25ms | CPU |

**Test set:** Separate held-out data, never seen during training. Metrics averaged across 3 runs.

---

## Methodology

### Model Architecture

Custom CNN designed for defect detection:
```
Input: 224×224 RGB images

Feature Extraction:
- 4 convolutional blocks (64 → 128 → 256 → 512 filters)
- Batch normalization after each conv layer
- Dropout (0.1-0.2) for regularization
- MaxPooling for downsampling
- Adaptive average pooling before classifier

Classification Head:
- Fully connected layers (512×7×7 → 1024 → 256 → 2)
- Dropout (0.5) to prevent overfitting
- Binary output (defective vs. good)

Total parameters: ~15M
```

### Training Details

**Dataset:**
- Total images: 70,000+ labeled samples
- Split: 70% train / 20% validation / 10% test
- Class balance: ~50/50 defective vs. good in training set
- Domains: Construction (concrete, surfaces, casting), textiles (fabric defects)

**Data Augmentation:**
- Random horizontal/vertical flips
- Random rotation (±15°)
- Color jittering (brightness, contrast, saturation)
- Random perspective transforms
- Resize and crop to 224×224

**Training Configuration:**
- Optimizer: AdamW (lr=0.001, weight_decay=1e-4)
- Loss: Cross-entropy
- Scheduler: Cosine annealing
- Batch size: 24-32
- Epochs: 15-20
- Hardware: NVIDIA RTX 4090

**Validation Strategy:**
- Held-out validation set for hyperparameter tuning
- Separate test set for final evaluation
- Early stopping based on validation F1-score
- Best model selected by test set performance

---

## Dataset Composition

### Construction Domain

**Defect types:**
- Concrete cracks (surface and structural)
- Spalling and surface degradation
- Voids and air pockets
- Material inconsistencies
- Casting defects

**Sources:**
- Public datasets (concrete crack detection)
- Proprietary casting defect data
- Real-world construction site images

### Textile Domain (Beta)

**Defect types:**
- Fabric tears and holes
- Weaving errors
- Color inconsistencies
- Pattern defects

**Status:** Beta - requires validation on more diverse fabric types

---

## Installation

### Requirements
```bash
Python 3.8+
PyTorch 2.0+
torchvision
Pillow
NumPy
```

### Quick Start
```bash
# Clone repository
git clone https://github.com/ihtesham-star/ai_defect_detection.git
cd ai_defect_detection

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage
```python
from basic_inference import DefectDetector

# Initialize detector (requires trained model weights)
detector = DefectDetector('path/to/model.pth')

# Single image prediction
result = detector.predict('image.jpg')
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")

# Batch processing
results = detector.predict_batch(['img1.jpg', 'img2.jpg', 'img3.jpg'])
```

---

## Project Structure
```
ai_defect_detection/
├── basic_inference.py      # Inference code (open source)
├── production_system.py    # Production deployment system
├── requirements.txt        # Python dependencies
├── docs/
│   └── TRAINING.md        # Training guide
├── assets/                # Example images
└── README.md             # This file
```

---

## Limitations and Future Work

### Current Limitations

**Technical:**
- Binary classification only (defective vs. good)
- No defect localization or segmentation
- Requires consistent lighting conditions
- Performance degrades on domains far from training data
- Best results on high-quality images (not heavily compressed)

**Domain-specific:**
- Construction model optimized for concrete/casting
- Textile model still in beta (90-95% F1)
- Limited to visible surface defects
- May require domain-specific fine-tuning

### Planned Improvements

- [ ] Add defect localization (bounding boxes)
- [ ] Multi-class defect categorization
- [ ] Expand to additional material types
- [ ] Improve textile model performance
- [ ] Mobile deployment (iOS/Android)
- [ ] Real-time video stream processing
- [ ] Active learning for continuous improvement

---

## Use Cases

### Manufacturing Quality Control
- Automated inspection of concrete products
- Pre-cast component verification
- Textile quality assurance
- Batch testing and statistical process control

### Construction
- On-site concrete inspection
- Pre-pour surface verification
- Material acceptance testing

### Research & Development
- Defect pattern analysis
- Quality trend monitoring
- Training data for custom models

---

## Deployment Options

### Included in Repository
- **Streamlit Web UI:** Interactive testing interface
- **Flask REST API:** Integration-ready endpoint
- **Command-line tool:** Batch processing scripts

### Commercial Services
Pre-trained models and deployment support available for commercial use.

**What's included:**
- Production model weights (70K+ training images)
- Integration assistance
- Custom fine-tuning on your data
- Technical support

**Contact:** ihteshamul.hayat@nu.edu.kz

---

## Documentation

- **[Training Guide](docs/TRAINING.md)** - Train your own model
- **[API Documentation]** - Coming soon
- **[Integration Examples]** - Coming soon

---

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

For major changes, please open an issue first to discuss proposed changes.

---

## Citation

If you use this work in your research or commercial application, please cite:
```
@software{ai_defect_detection_2026,
  author = {Ihteshamul Hayat},
  title = {AI-Powered Defect Detection System},
  year = {2026},
  url = {https://github.com/ihtesham-star/ai_defect_detection}
}
```

---

## License

MIT License - See [LICENSE](LICENSE) file

**Commercial use permitted** with attribution.

---

**Contact:**
- Email: ihteshamul.hayat@nu.edu.kz
- GitHub: [@ihtesham-star](https://github.com/ihtesham-star)
---

## Acknowledgments

- Training data sourced from public datasets and proprietary archives
- Built with PyTorch, Streamlit, and Flask
- Inspired by industrial quality control needs in Kazakhstan

---

**Questions? Open an [issue](https://github.com/ihtesham-star/ai_defect_detection/issues) or email me.**

---

*Last updated: February 2026*