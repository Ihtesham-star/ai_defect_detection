\# Training Guide



\## Training Your Own Model



\### Requirements

\- \*\*Dataset:\*\* 10,000+ labeled images (minimum for decent results)

\- \*\*Recommended:\*\* 50,000+ images for production quality

\- \*\*Hardware:\*\* NVIDIA GPU with 8GB+ VRAM

\- \*\*Time:\*\* 3-7 days of continuous training

\- \*\*Storage:\*\* 500GB+ for dataset and checkpoints



\### Dataset Structure

```

dataset/

├── train/

│   ├── defective/

│   │   ├── img001.jpg

│   │   └── ...

│   └── good/

│       ├── img001.jpg

│       └── ...

└── val/

&nbsp;   ├── defective/

&nbsp;   └── good/

```



\### Training Command

```bash

python train.py \\

&nbsp;   --data ./dataset \\

&nbsp;   --epochs 50 \\

&nbsp;   --batch-size 32 \\

&nbsp;   --learning-rate 0.001 \\

&nbsp;   --save-dir ./models

```



\### Expected Training Time

\- 10K images: 1-2 days

\- 50K images: 3-5 days  

\- 70K images: 5-7 days



---



\## OR Save Time: Get Pre-Trained Model



\*\*Skip weeks of work and get our production-ready model:\*\*



✅ Trained on 70,000+ images  

✅ 98%+ accuracy on construction  

✅ Ready to deploy immediately  

✅ Commercial license included  



\*\*Price:\*\* $299 one-time OR $49/month



\*\*Contact:\*\* ihteshamul.hayat@nu.edu.kz

