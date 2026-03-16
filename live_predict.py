import serial
import numpy as np
import tensorflow as tf
import time

# --- CONFIG ---
COM_PORT = 'COM5'  # Update this to your port
BAUD_RATE = 115200
MODEL_PATH = "silent_speech_model.h5"
LABEL_PATH = "labels.npy"

# Load model and setup serial
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    labels = np.load(LABEL_PATH)
    ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print(f"✅ System Ready. Connected to {COM_PORT}")
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

def process_and_predict(signal_list):
    data = np.array(signal_list, dtype=float)
    # Use the same normalization as your prepare script
    data = (data - np.mean(data)) / (np.std(data) + 1e-8)
    
    # Resize to 400 (The model expects exactly 400)
    if len(data) > 400:
        data = data[:400]
    else:
        data = np.pad(data, (0, 400 - len(data)))
        
    data = data.reshape(1, 400, 1)
    prediction = model.predict(data, verbose=0)
    idx = np.argmax(prediction)
    return labels[idx], prediction[0][idx]

print("\n" + "="*30)
print(" SILENT SPEECH LIVE CONTROL ")
print("="*30)

try:
    while True:
        input("\n🎤 Press [ENTER] to prepare...")
        
        # Flush the buffer to fix "Incomplete Sample" errors
        ser.reset_input_buffer()
        ser.write(b's')
        
        # UI Countdown
        print("Ready in: 3...", end="\r")
        time.sleep(0.5)
        print("Ready in: 2...", end="\r")
        time.sleep(0.5)
        print("Ready in: 1...", end="\r")
        time.sleep(0.5)
        print(">>> START SPEAKING NOW! <<<")
        
        raw_samples = []
        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line == "DONE":
                break
            try:
                raw_samples.append(float(line))
            except ValueError:
                continue
        
        print("Recording Finished. Processing...")

        if len(raw_samples) >= 100:
            # We take the middle section of the 600 samples to ensure the word is captured
            # This handles if you start a little late or end early
            center = len(raw_samples) // 2
            start_idx = max(0, center - 200)
            end_idx = min(len(raw_samples), center + 200)
            final_clip = raw_samples[start_idx:end_idx]

            word, confidence = process_and_predict(final_clip)
            
            # Formatting the output nicely
            color = "\033[1;32m" if confidence > 0.6 else "\033[1;33m"
            print(f"\nResult: {color}{word.upper()}\033[0m ({confidence*100:.1f}%)")
        else:
            print(f"⚠ Error: Sample too short ({len(raw_samples)} pts). Check wiring.")

except KeyboardInterrupt:
    ser.close()
    print("\nSerial connection closed. Goodbye!")
