# backend/train.py

from utils import preprocess_and_train

if __name__ == '__main__':
    print("🔧 Starting training using courses.csv and interactions.csv...")
    try:
        preprocess_and_train()
        print("✅ Training complete. All model files saved in backend/models/")
    except Exception as e:
        print("❌ Error during training:", e)