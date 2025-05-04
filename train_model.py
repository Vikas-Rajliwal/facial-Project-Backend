from sklearn.ensemble import RandomForestClassifier
from joblib import dump
import numpy as np
from model_utils import load_npz_data
import os
from sklearn.metrics import classification_report, accuracy_score

def train_and_save_model():
    # Load training and test data
    try:
        print("Loading training data...")
        train_images, train_labels = load_npz_data("train_data.npz")
        print(f"Training data loaded: {train_images.shape} images, {len(train_labels)} labels")
        
        print("Loading test data...")
        test_images, test_labels = load_npz_data("test_data.npz")
        print(f"Test data loaded: {test_images.shape} images, {len(test_labels)} labels")
    except FileNotFoundError as e:
        print(f"❌ Error: {str(e)}")
        print("Please ensure both train_data.npz and test_data.npz exist in the current directory.")
        return False
    
    # Preprocess images - flatten and normalize
    print("\nPreprocessing data...")
    train_images_flat = train_images.reshape(len(train_images), -1).astype('float32') / 255.0
    test_images_flat = test_images.reshape(len(test_images), -1).astype('float32') / 255.0
    
    # Train the Random Forest classifier
    print("Training model...")
    emotion_classifier = RandomForestClassifier(
        n_estimators=10,
        random_state=20,
        n_jobs=-1  # Use all available CPU cores
    )
    emotion_classifier.fit(train_images_flat, train_labels)
    
    # Evaluate the model
    print("\nEvaluating model performance...")
    train_accuracy = emotion_classifier.score(train_images_flat, train_labels)
    test_accuracy = emotion_classifier.score(test_images_flat, test_labels)
    
    print(f"\nTraining Accuracy: {train_accuracy:.2%}")
    print(f"Test Accuracy: {test_accuracy:.2%}")
    
    # Detailed classification report
    test_predictions = emotion_classifier.predict(test_images_flat)
    print("\nDetailed Classification Report:")
    print(classification_report(test_labels, test_predictions, target_names=[
        'Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'
    ]))
    
    # Save the trained model
    model_path = "emotion_classifier.joblib"
    dump(emotion_classifier, model_path)
    print(f"\n✅ Model trained and saved as {model_path}")
    print(f"   Full path: {os.path.abspath(model_path)}")
    return True

if __name__ == "__main__":
    train_and_save_model() 