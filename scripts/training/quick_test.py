from inference import load_trained_model, predict_single_sample
import numpy as np

def create_challenging_test_cases():
    
    try:
        # Load your training data
        X = np.load("/Users/jemharoldcentino/Documents/GitHub/Dory/features/X.npy", allow_pickle=True)
        y = np.load("/Users/jemharoldcentino/Documents/GitHub/Dory/features/y.npy", allow_pickle=True)
        
        # Load trained model
        model, encoder = load_trained_model()
        
        print(f"🔬 ADVANCED TESTING - Challenging Scenarios")
        print(f"="*60)
        
        # Get sample shapes
        sample_shape = X[0].shape
        
        # Test 1: Very noisy wake sample
        print(f"\n1️⃣ NOISY WAKE WORD TEST")
        print("-" * 30)
        
        wake_indices = np.where(y == 'wake')[0]
        if len(wake_indices) > 0:
            clean_wake = X[wake_indices[0]]
            
            # Add increasing levels of noise
            noise_levels = [0.1, 0.3, 0.5, 0.7, 1.0]
            
            for noise_level in noise_levels:
                noise = np.random.normal(0, noise_level * np.std(clean_wake), clean_wake.shape)
                noisy_wake = clean_wake + noise
                
                prediction, confidence, _ = predict_single_sample(model, encoder, noisy_wake)
                
                status = "✅" if prediction == 'wake' else "❌"
                print(f"   {status} Noise level {noise_level:.1f}: {prediction} ({confidence:.1%})")
        
        # Test 2: Mixed samples (average of different classes)
        print(f"\n2️⃣ MIXED SAMPLE TEST")
        print("-" * 30)
        
        classes = ['noise', 'not_wake', 'wake']
        for i, class1 in enumerate(classes):
            for class2 in classes[i+1:]:
                # Get samples from each class
                idx1 = np.where(y == class1)[0]
                idx2 = np.where(y == class2)[0]
                
                if len(idx1) > 0 and len(idx2) > 0:
                    sample1 = X[idx1[0]]
                    sample2 = X[idx2[0]]
                    
                    # Create 50/50 mix
                    mixed_sample = (sample1 + sample2) / 2
                    
                    prediction, confidence, probabilities = predict_single_sample(
                        model, encoder, mixed_sample
                    )
                    
                    print(f"   🔀 {class1} + {class2} → {prediction} ({confidence:.1%})")
                    
                    # Show confusion
                    prob1 = probabilities[encoder.classes_.tolist().index(class1)]
                    prob2 = probabilities[encoder.classes_.tolist().index(class2)]
                    print(f"      {class1}: {prob1:.1%}, {class2}: {prob2:.1%}")
        
        # Test 3: Scaled samples (very quiet vs very loud)
        print(f"\n3️⃣ AMPLITUDE TEST")
        print("-" * 30)
        
        for class_name in classes:
            class_indices = np.where(y == class_name)[0]
            if len(class_indices) > 0:
                original_sample = X[class_indices[0]]
                
                # Test different amplitudes
                scales = [0.01, 0.1, 2.0, 5.0]  # Very quiet to very loud
                
                print(f"   {class_name.upper()} amplitude tests:")
                for scale in scales:
                    scaled_sample = original_sample * scale
                    prediction, confidence, _ = predict_single_sample(
                        model, encoder, scaled_sample
                    )
                    
                    status = "✅" if prediction == class_name else "❌"
                    print(f"      {status} Scale {scale:4.2f}x: {prediction} ({confidence:.1%})")
        
        # Test 4: Edge case - all zeros and all ones
        print(f"\n4️⃣ EDGE CASE TEST")
        print("-" * 30)
        
        # All zeros (silence)
        zero_sample = np.zeros(sample_shape)
        prediction, confidence, probabilities = predict_single_sample(model, encoder, zero_sample)
        print(f"   🔇 All zeros: {prediction} ({confidence:.1%})")
        
        # All ones (maximum signal)
        ones_sample = np.ones(sample_shape)
        prediction, confidence, probabilities = predict_single_sample(model, encoder, ones_sample)
        print(f"   📢 All ones: {prediction} ({confidence:.1%})")
        
        # Random uniform noise
        uniform_noise = np.random.uniform(-1, 1, sample_shape)
        prediction, confidence, probabilities = predict_single_sample(model, encoder, uniform_noise)
        print(f"   🎲 Uniform noise: {prediction} ({confidence:.1%})")
        
        # Test 5: Confidence threshold analysis
        print(f"\n5️⃣ CONFIDENCE ANALYSIS")
        print("-" * 30)
        
        confidence_scores = []
        true_labels = []
        predictions = []
        
        # Test all samples and collect confidence scores
        for i in range(min(20, len(X))):  # Test first 20 samples
            test_sample = X[i]
            true_label = y[i]
            
            prediction, confidence, _ = predict_single_sample(model, encoder, test_sample)
            
            confidence_scores.append(confidence)
            true_labels.append(true_label)
            predictions.append(prediction)
        
        # Calculate statistics
        correct_confidences = [conf for i, conf in enumerate(confidence_scores) 
                             if predictions[i] == true_labels[i]]
        incorrect_confidences = [conf for i, conf in enumerate(confidence_scores) 
                               if predictions[i] != true_labels[i]]
        
        if correct_confidences:
            print(f"   ✅ Correct predictions confidence:")
            print(f"      Average: {np.mean(correct_confidences):.1%}")
            print(f"      Min: {np.min(correct_confidences):.1%}")
            print(f"      Max: {np.max(correct_confidences):.1%}")
        
        if incorrect_confidences:
            print(f"   ❌ Incorrect predictions confidence:")
            print(f"      Average: {np.mean(incorrect_confidences):.1%}")
            print(f"      Min: {np.min(incorrect_confidences):.1%}")
            print(f"      Max: {np.max(incorrect_confidences):.1%}")
        else:
            print(f"   🎉 No incorrect predictions in test set!")
        
        # Suggest optimal confidence threshold
        if correct_confidences:
            min_correct_conf = np.min(correct_confidences)
            suggested_threshold = min_correct_conf * 0.95  # 95% of minimum correct confidence
            print(f"\n💡 SUGGESTED SETTINGS:")
            print(f"   Confidence threshold for wake word: {suggested_threshold:.1%}")
            print(f"   This would catch all correct predictions while minimizing false positives")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def test_wake_word_thresholds():
    """Test different confidence thresholds for wake word detection"""
    
    try:
        # Load data and model
        X = np.load("/Users/jemharoldcentino/Documents/GitHub/Dory/features/X.npy", allow_pickle=True)
        y = np.load("/Users/jemharoldcentino/Documents/GitHub/Dory/features/y.npy", allow_pickle=True)
        model, encoder = load_trained_model()
        
        print(f"\n🎯 WAKE WORD THRESHOLD ANALYSIS")
        print(f"="*50)
        
        # Test different thresholds
        thresholds = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
        
        wake_samples = X[y == 'wake']
        non_wake_samples = X[y != 'wake']
        
        print(f"Testing with {len(wake_samples)} wake samples and {len(non_wake_samples)} non-wake samples")
        print(f"\nThreshold | Wake Detected | False Positives | False Negatives")
        print("-" * 60)
        
        for threshold in thresholds:
            wake_detected = 0
            false_positives = 0
            false_negatives = 0
            
            # Test wake samples
            for sample in wake_samples:
                prediction, confidence, _ = predict_single_sample(model, encoder, sample)
                if confidence >= threshold and prediction == 'wake':
                    wake_detected += 1
                elif confidence < threshold or prediction != 'wake':
                    false_negatives += 1
            
            # Test non-wake samples
            for sample in non_wake_samples:
                prediction, confidence, _ = predict_single_sample(model, encoder, sample)
                if confidence >= threshold and prediction == 'wake':
                    false_positives += 1
            
            # Calculate rates
            detection_rate = wake_detected / len(wake_samples) * 100 if len(wake_samples) > 0 else 0
            fp_rate = false_positives / len(non_wake_samples) * 100 if len(non_wake_samples) > 0 else 0
            
            print(f"{threshold:8.2f} | {detection_rate:11.1f}% | {fp_rate:13.1f}% | {false_negatives:13d}")
        
        print(f"\n💡 Recommended threshold: 0.90 (good balance of detection vs false positives)")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    create_challenging_test_cases()
    test_wake_word_thresholds()