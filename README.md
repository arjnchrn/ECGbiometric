# ECG Biometric Authentication for IoT Edge Devices


This project implements a CNN-based ECG biometric authentication system with complexity optimization techniques for IoT edge device deployment. The system achieves:
- **99.63%** baseline authentication accuracy (original weights)
- **98.88%** optimized accuracy with reduced complexity (approximate weights)
- **91.06%** accuracy with maximum complexity reduction (binary weights)

## Features

### 1. Data Preprocessing
- Automatic download of PTB Diagnostic ECG Database (290 subjects)
- ECG signal resampling to 200 Hz
- R-peak detection using biosppy
- QRS complex extraction (1-second windows, 200 samples each)
- Random selection of 20 complexes per subject

### 2. CNN Model Architecture
Following the paper's specification:
- **Conv1D Layer 1**: 32 filters, kernel size 5, ReLU activation
- **MaxPooling1D**: Pool size 2
- **Dropout**: 20%
- **Conv1D Layer 2**: 64 filters, kernel size 5, ReLU activation
- **MaxPooling1D**: Pool size 2
- **Dropout**: 20%
- **Conv1D Layer 3**: 64 filters, kernel size 5, ReLU activation
- **MaxPooling1D**: Pool size 2
- **Dropout**: 20%
- **Flatten**
- **Dense**: 128 units, ReLU activation
- **Dropout**: 50%
- **Output Dense**: num_subjects units, Softmax activation

### 3. Feature Extraction
- Trained CNN with fully-connected layers for classification
- Post-training: FC layers are discarded
- Convolutional layers are kept as feature extractors
- High-dimensional feature vectors used for authentication

### 4. Authentication System
**User/Intruder Evaluation Protocol:**
- Dataset randomly divided into "enrolled users" and "intruders"
- Feature vectors stored in database for enrolled users
- Authentication via Euclidean distance matching:
  ```
  d(x, y) = √[(x₁-y₁)² + (x₂-y₂)² + ... + (xₙ-yₙ)²]
  ```
- Threshold-based decision: Accept if distance < threshold

**Metrics:**
- Authentication Accuracy
- False Accept Rate (FAR)
- False Reject Rate (FRR)
- Equal Error Rate (EER)
- ROC Curve and AUC

### 5. Complexity Optimization

#### Binary Weights
- Apply SIGN function: w → {+1, -1}
- Replaces floating-point multiplications with 1-bit inversions
- Minimal CPU cycles but reduced accuracy

#### Approximate Weights
- Exponential representation: w = 2^a + 2^b + ...
- Replaces multiplications with bit shifts and additions
- Parameter n controls approximation accuracy (n=1, 2, 3)
- Better accuracy-complexity trade-off

#### CPU Cycle Analysis (AMD K7 Architecture)
- **Multiplication**: 3 cycles
- **Inversion**: 1 cycle
- **Bit Shift**: 1 cycle
- **Addition**: 1 cycle

## Installation


1. **Extract the zip file** to your desired location

2. **Install Python 3.8 or higher** (if not already installed)
   - Download from: https://www.python.org/downloads/

3. **Open a terminal/command prompt** and navigate to the project folder:
   ```bash
   cd path/to/extracted/project
   ```

4. **Install required dependencies**:
   ```bash
   pip install tensorflow numpy scipy wfdb scikit-learn matplotlib plotly biosppy pandas peakutils streamlit
   ```
   
   **Optional:** Create a `requirements.txt` file with the following content for easier installation:
   ```
   tensorflow>=2.10.0
   numpy>=1.21.0
   scipy>=1.7.0
   wfdb>=4.0.0
   scikit-learn>=1.0.0
   matplotlib>=3.5.0
   plotly>=5.0.0
   biosppy>=0.8.0
   pandas>=1.3.0
   peakutils>=1.3.0
   streamlit>=1.20.0
   ```
   
   Then install with:
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the Streamlit app**:
   ```bash
   streamlit run app.py --server.port 5000
   ```

6. **Open your browser** and go to:
   ```
   http://localhost:5000
   ```


## Usage

Navigate through the 9 interactive sections:

### Step 1: Data Preprocessing
1. Click "Load/Process PTB Database"
2. View sample ECG complexes and dataset statistics
3. Data is automatically cached for future use

### Step 2: Model Training
1. Configure epochs (default: 20) and batch size (default: 16)
2. Click "Train CNN Model"
3. View real-time training progress
4. Examine training history charts

### Step 3: Feature Extraction
1. Click "Create Feature Extractor"
2. View feature extractor architecture
3. See which layers were discarded vs. kept

### Step 4: Authentication Testing
1. Set enrolled user ratio (default: 50%)
2. Click "Evaluate Authentication System"
3. View accuracy, FAR, FRR, EER metrics
4. Examine ROC curve

### Step 5: Optimization & Results
1. Click "Evaluate All Optimization Variants"
2. Compare accuracy vs. complexity trade-offs
3. View CPU cycle comparisons
4. Examine operation breakdowns
5. Compare with paper's reported results

### Step 6: Live ECG Authentication
1. Upload your own ECG signal (CSV, TXT, or NPY format)
2. Specify the original sampling rate
3. Click "Process & Authenticate"
4. View R-peak detection and authentication results
5. See which enrolled user (if any) matches your signal

### Step 7: Advanced Optimizations
1. **TensorFlow Lite Quantization**: Convert model to INT8 or FLOAT16
2. **Manual Pruning**: Remove weights below a threshold percentile
3. Compare model sizes and complexity reductions

### Step 8: Cross-Validation
1. Configure number of folds (K) and enrolled user ratio
2. Run cross-validation with different train/test splits
3. View statistical analysis (mean, std, min, max accuracy)
4. Examine performance consistency across folds

### Step 9: Model Export for IoT
1. Export trained models in multiple formats:
   - TensorFlow Lite (.tflite)
   - ONNX (.onnx)
   - Keras (.h5)
   - C++ header files (.h) for ARM Cortex-M microcontrollers
2. Download files for deployment on edge devices

## Architecture

```
data_preprocessing.py    - PTB database handling, signal processing
cnn_model.py            - CNN architecture and feature extraction
authentication_system.py - Euclidean distance matching, evaluation
optimization.py         - Binary/approximate weight transformations
app.py                 - Streamlit interactive dashboard
```


### Classification → Feature Extraction → Authentication

1. **Training Phase**: CNN is trained as a multi-class classifier (290 subjects)
2. **Feature Extraction**: Remove FC layers, keep convolutional layers
3. **Enrollment**: Store feature vectors of enrolled users in database
4. **Authentication**: Calculate Euclidean distance between test features and enrolled features
5. **Decision**: Accept if minimum distance < threshold, reject otherwise


> "To generate high dimensional features of the ECG signals, we discarded fully-connected 
> layers of trained CNN structure and kept the convolution layers which contain information 
> of ECG signal."

This approach uses the CNN as a feature extractor rather than a classifier, enabling 
distance-based authentication suitable for continuous monitoring.


## Dataset

**PTB Diagnostic ECG Database:**
- 549 records from 290 subjects
- Ages 17-87 years
- 15 simultaneously measured signals
- Original sampling: 1000 Hz
- Resampled to: 200 Hz
- Available at: https://physionet.org/content/ptbdb/1.0.0/

## Troubleshooting

### Common Issues

**1. TensorFlow Version Compatibility**
- The application is compatible with TensorFlow 2.10+
- If you encounter attribute errors, ensure you have the latest compatible version:
  ```bash
  pip install --upgrade tensorflow
  ```

**2. Database Download Fails**
- The PTB database download requires a stable internet connection
- If download fails, the app will retry automatically
- You can manually download from: https://physionet.org/content/ptbdb/1.0.0/

**3. Memory Issues During Training**
- Training the CNN requires ~2-4 GB of RAM
- If you encounter memory errors, try reducing the batch size in Step 2
- Close other applications to free up memory

**4. File Upload Issues (Step 6)**
- Supported formats: CSV, TXT (whitespace or comma-delimited), NPY
- Ensure your ECG file contains a single-channel signal
- If upload fails, check that the file is not corrupted

**5. Missing Dependencies**
- If you get import errors, ensure all dependencies are installed:
  ```bash
  pip install tensorflow numpy scipy wfdb scikit-learn matplotlib plotly biosppy pandas peakutils streamlit
  ```
