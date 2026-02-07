# 🔍 AI-Powered Defect Detection System

**Production-ready AI for automated quality control in construction and manufacturing.**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 What This Does

Automatically detects defects in:
- ✅ **Construction Materials** (concrete, surfaces) - **Production Ready**
- 🧪 **Textiles & Fabrics** - **Beta**

**Performance:**
- **Speed:** 17-37ms per image
- **Accuracy:** 98%+ on construction materials
- **Throughput:** 100+ images per minute

---

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/ihtesham-star/ai_defect_detection.git
cd ai_defect_detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Get Pre-trained Models

**This repository contains the code architecture. To use the system, you need the trained model weights:**

**Option A - Free Trial:**
- Email: ihteshamul.hayat@nu.edu.kz
- Subject: "Request Free Trial Model"
- Get access to sample model for testing

**Option B - Full Production Model:**
- Construction-grade model trained on 70,000+ images
- See pricing below

---

## 📁 Project Structure
```
ai_defect_detection/
├── basic_inference.py      # Open-source inference code
├── production_system.py    # Production system (requires trained model)
├── requirements.txt        # Python dependencies
├── docs/
│   └── TRAINING.md        # Guide for training your own model
├── assets/                # Demo screenshots
└── README.md             # You are here
```

---

## 💡 How It Works

### Architecture
- **Model:** Custom CNN with batch normalization
- **Input:** 224×224 RGB images
- **Output:** Binary classification (Defective/Good) + confidence score
- **Framework:** PyTorch 2.0+

### Usage Example
```python
from basic_inference import DefectDetector

# Initialize (requires trained model)
detector = DefectDetector('path/to/model.pth')

# Single image
result = detector.predict('image.jpg')
print(f"Status: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")

# Batch processing
results = detector.predict_batch(['img1.jpg', 'img2.jpg', 'img3.jpg'])
```

---

## 📊 What You Get

### ✅ Open Source (Free)
- Model architecture code
- Training documentation
- Basic inference scripts
- MIT License

### 💼 Production Model (Paid)
- Pre-trained weights (70,000+ images)
- Web interface (Streamlit + Flask API)
- Batch processing
- Priority support

---

## 💰 Pricing

| Tier | Price | What's Included |
|------|-------|-----------------|
| **DIY** | Free | Code only, train your own model |
| **Pre-trained Model** | $299 one-time OR $49/month | Trained model weights + basic support |
| **Hosted Solution** | $99-299/month | SaaS platform, API access, updates |
| **Enterprise** | Custom | Custom training, integration, SLA |

**🎁 Special Offer:** First 10 customers get 50% off!

---

## 🎯 Use Cases

### Construction ✅ Production Ready
- Concrete surface inspection
- Crack detection
- Material quality control
- Pre-pour verification

### Textiles 🧪 Beta
- Fabric defect detection
- Weaving errors
- Color inconsistencies
- Quality grading

---

## 📈 Performance

| Domain | Accuracy | Speed | Status |
|--------|----------|-------|--------|
| Construction | 98-100% | 20ms | ✅ Production |
| Textiles | 90-95% | 25ms | 🧪 Beta |

**Training Dataset:**
- 70,000+ labeled images
- Multiple defect types
- Real-world conditions
- Balanced classes

---

## 🛠️ Technical Requirements
```
Python 3.8+
PyTorch 2.0+
torchvision
Pillow
NumPy
```

**Optional (for production system):**
```
Flask
Streamlit
pandas
```

---

## 📖 Documentation

- **[Training Guide](docs/TRAINING.md)** - How to train your own model
- **API Documentation** - Coming soon
- **Integration Examples** - Coming soon

---

## 🤝 Get Started

**Interested in using this for your business?**

1. **Free Consultation:** Email me to discuss your use case
2. **Free Trial:** Test on your sample images
3. **Pilot Program:** 30-day money-back guarantee

**Contact:**
- Email: ihteshamul.hayat@nu.edu.kz
- GitHub Issues: For technical questions

---

## ⚠️ Important Notes

### Construction Materials ✅
- Fully validated and production-ready
- 98%+ accuracy on unseen data
- Recommended for automated decision-making

### Textiles 🧪
- Beta quality - suitable for assisted inspection
- Requires human review for critical decisions
- Continuous improvement in progress

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file

**Commercial Use:** Allowed with proper attribution

---

## 🙏 Credits

Developed by **Sham** (Research Assistant, CEMRR, Nazarbayev University)

Trained on curated datasets from:
- Construction quality control archives
- Textile manufacturing facilities
- Public defect detection benchmarks

---

## 🔮 Roadmap

- [ ] Mobile app (iOS/Android)
- [ ] Real-time video processing
- [ ] Additional material types
- [ ] Cloud API deployment
- [ ] Multi-language support

---

## ❓ FAQ

**Q: Can I train my own model?**  
A: Yes! See [docs/TRAINING.md](docs/TRAINING.md) for instructions.

**Q: Do I need GPU?**  
A: For inference: No. CPU works fine (20-40ms).  
For training: Yes, GPU recommended.

**Q: What image formats are supported?**  
A: JPG, PNG, BMP, TIFF

**Q: Can I use this commercially?**  
A: Yes, with paid license or by training your own model.

---

**⭐ If you find this useful, please star the repository!**

**🐛 Found a bug? [Open an issue](https://github.com/ihtesham-star/ai_defect_detection/issues)**

---

*Last updated: February 2026*