import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import pickle
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

from data_preprocessing import PTBDataPreprocessor
from cnn_model import ECGAuthenticationCNN
from authentication_system import ECGAuthenticator
from optimization import WeightOptimizer

st.set_page_config(page_title="ECG Biometric Authentication", layout="wide")

st.title("ECG Biometric Authentication for IoT Edge Devices")
st.markdown("### Recreation of: *Low Complexity ECG Biometric Authentication for IoT Edge Devices* (Wang et al., 2020)")

@st.cache_data
def load_or_process_data():
    """Load preprocessed data or process if not available"""
    preprocessor = PTBDataPreprocessor(target_fs=200, window_size=1.0, num_complexes=20)
    
    X, y, subjects = preprocessor.load_processed_data()
    
    if X is None:
        st.info("Processing PTB database for the first time. This may take several minutes...")
        
        if not os.path.exists(preprocessor.data_dir):
            preprocessor.download_ptb_database()
        
        X, y, subjects = preprocessor.load_and_preprocess_data(max_subjects=290)
        
        if X is not None and len(X) > 0:
            preprocessor.save_processed_data(X, y, subjects)
    
    return X, y, subjects

def create_train_test_split(X, y):
    """Create train/test split"""
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def plot_ecg_sample(X, y, subjects, num_samples=5):
    """Plot sample ECG complexes"""
    fig = make_subplots(
        rows=num_samples, cols=1,
        subplot_titles=[f"Subject: {subjects[y[i]]}" for i in range(num_samples)]
    )
    
    for i in range(num_samples):
        fig.add_trace(
            go.Scatter(x=list(range(200)), y=X[i].flatten(), mode='lines', name=subjects[y[i]]),
            row=i+1, col=1
        )
    
    fig.update_layout(height=800, showlegend=False, title_text="Sample QRS Complexes (200 samples @ 200 Hz)")
    fig.update_xaxes(title_text="Sample Index")
    fig.update_yaxes(title_text="Amplitude")
    
    return fig

def plot_training_history(history):
    """Plot training history"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Model Accuracy', 'Model Loss']
    )
    
    fig.add_trace(
        go.Scatter(y=history.history['accuracy'], name='Train Accuracy', mode='lines'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(y=history.history['val_accuracy'], name='Val Accuracy', mode='lines'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(y=history.history['loss'], name='Train Loss', mode='lines'),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(y=history.history['val_loss'], name='Val Loss', mode='lines'),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="Accuracy", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=2)
    fig.update_layout(height=400, title_text="Training Performance")
    
    return fig

def plot_roc_curve(fpr, tpr, roc_auc):
    """Plot ROC curve"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {roc_auc:.4f})',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title='ROC Curve - Authentication Performance',
        xaxis_title='False Positive Rate (FAR)',
        yaxis_title='True Positive Rate (1 - FRR)',
        height=500,
        width=600
    )
    
    return fig

def plot_complexity_comparison(optimization_results):
    """Plot accuracy vs complexity trade-off"""
    variants = list(optimization_results.keys())
    accuracies = [optimization_results[v]['accuracy'] * 100 for v in variants]
    cpu_cycles = [optimization_results[v]['cpu_cycles'] for v in variants]
    
    variant_labels = {
        'original': 'Original',
        'binary': 'Binary',
        'approx_n1': 'Approx (n=1)',
        'approx_n2': 'Approx (n=2)',
        'approx_n3': 'Approx (n=3)'
    }
    
    labels = [variant_labels.get(v, v) for v in variants]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=cpu_cycles,
        y=accuracies,
        mode='markers+text',
        marker=dict(size=15, color=accuracies, colorscale='Viridis', showscale=True),
        text=labels,
        textposition='top center',
        textfont=dict(size=10)
    ))
    
    fig.update_layout(
        title='Accuracy vs. CPU Cycles Trade-off',
        xaxis_title='CPU Cycles',
        yaxis_title='Accuracy (%)',
        height=500,
        width=800
    )
    
    return fig

def plot_operations_comparison(optimization_results):
    """Plot operations breakdown for different variants"""
    variants = list(optimization_results.keys())
    
    variant_labels = {
        'original': 'Original',
        'binary': 'Binary',
        'approx_n1': 'Approx (n=1)',
        'approx_n2': 'Approx (n=2)',
        'approx_n3': 'Approx (n=3)'
    }
    
    labels = [variant_labels.get(v, v) for v in variants]
    
    mult = [optimization_results[v]['operations']['multiplication'] for v in variants]
    inv = [optimization_results[v]['operations']['inversion'] for v in variants]
    shift = [optimization_results[v]['operations']['bit_shift'] for v in variants]
    add = [optimization_results[v]['operations']['addition'] for v in variants]
    
    fig = go.Figure(data=[
        go.Bar(name='Multiplication', x=labels, y=mult),
        go.Bar(name='Inversion', x=labels, y=inv),
        go.Bar(name='Bit Shift', x=labels, y=shift),
        go.Bar(name='Addition', x=labels, y=add)
    ])
    
    fig.update_layout(
        title='Number of Operations by Weight Variant',
        xaxis_title='Weight Variant',
        yaxis_title='Number of Operations',
        barmode='group',
        height=500
    )
    
    return fig

st.sidebar.header("Navigation")
page = st.sidebar.radio("Select Page", [
    "1. Data Preprocessing",
    "2. Model Training",
    "3. Feature Extraction",
    "4. Authentication Testing",
    "5. Optimization & Results",
    "6. Live ECG Authentication",
    "7. Advanced Optimizations",
    "8. Cross-Validation",
    "9. Model Export"
])

if page == "1. Data Preprocessing":
    st.header("Step 1: Data Preprocessing")
    
    st.markdown("""
    This step preprocesses the PTB ECG database according to the paper:
    - Resample signals to 200 Hz
    - Detect R-peaks using biosppy
    - Extract 1-second QRS complexes (200 samples each)
    - Randomly select 20 complexes per subject
    - Process 290 subjects from the database
    """)
    
    if st.button("Load/Process PTB Database"):
        with st.spinner("Loading data..."):
            X, y, subjects = load_or_process_data()
            
            if X is not None:
                st.session_state['X'] = X
                st.session_state['y'] = y
                st.session_state['subjects'] = subjects
                
                st.success(f"✓ Data loaded successfully!")
                st.write(f"- **Total samples**: {len(X)}")
                st.write(f"- **Number of subjects**: {len(subjects)}")
                st.write(f"- **Input shape**: {X.shape}")
                st.write(f"- **Samples per complex**: {X.shape[1]}")
    
    if 'X' in st.session_state:
        st.subheader("Sample ECG Complexes")
        fig = plot_ecg_sample(st.session_state['X'], st.session_state['y'], 
                             st.session_state['subjects'], num_samples=5)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Dataset Statistics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Subjects", len(st.session_state['subjects']))
        col2.metric("Total Samples", len(st.session_state['X']))
        col3.metric("Samples/Subject", len(st.session_state['X']) // len(st.session_state['subjects']))

elif page == "2. Model Training":
    st.header("Step 2: CNN Model Training")
    
    st.markdown("""
    **CNN Architecture** (as per paper):
    - Conv1D (32 filters, kernel=5) + MaxPool + Dropout
    - Conv1D (64 filters, kernel=5) + MaxPool + Dropout
    - Conv1D (64 filters, kernel=5) + MaxPool + Dropout
    - Flatten + Dense(128) + Dropout + Dense(num_subjects)
    
    **Training Parameters**:
    - Optimizer: Adam (Gradient Descent)
    - Epochs: 100
    - Batch Size: 32
    """)
    
    if 'X' not in st.session_state:
        st.warning("Please load data from Step 1 first!")
    else:
        col1, col2 = st.columns(2)
        epochs = col1.number_input("Epochs", value=100, min_value=1, max_value=200)
        batch_size = col2.number_input("Batch Size", value=32, min_value=4, max_value=128)
        
        if st.button("Train CNN Model"):
            X = st.session_state['X']
            y = st.session_state['y']
            
            with st.spinner("Splitting data..."):
                X_train, X_test, y_train, y_test = create_train_test_split(X, y)
                st.session_state['X_train'] = X_train
                st.session_state['X_test'] = X_test
                st.session_state['y_train'] = y_train
                st.session_state['y_test'] = y_test
                
                st.info(f"Train set: {len(X_train)} samples | Test set: {len(X_test)} samples")
            
            with st.spinner("Training model... This may take a few minutes."):
                num_classes = len(np.unique(y))
                cnn = ECGAuthenticationCNN(input_shape=(200, 1), num_classes=num_classes)
                cnn.build_model()
                cnn.compile_model()
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                class StreamlitCallback(tf.keras.callbacks.Callback):
                    def on_epoch_end(self, epoch, logs=None):
                        progress_bar.progress((epoch + 1) / epochs)
                        status_text.text(f"Epoch {epoch + 1}/{epochs} - "
                                       f"Loss: {logs['loss']:.4f} - "
                                       f"Acc: {logs['accuracy']:.4f} - "
                                       f"Val Acc: {logs['val_accuracy']:.4f}")
                
                history = cnn.model.fit(
                    X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=0,
                    callbacks=[StreamlitCallback()]
                )
                
                st.session_state['cnn'] = cnn
                st.session_state['history'] = history
                
                final_acc = history.history['accuracy'][-1]
                final_val_acc = history.history['val_accuracy'][-1]
                
                st.success(f"✓ Training complete!")
                st.write(f"- **Final Training Accuracy**: {final_acc*100:.2f}%")
                st.write(f"- **Final Validation Accuracy**: {final_val_acc*100:.2f}%")
        
        if 'history' in st.session_state:
            st.subheader("Training History")
            fig = plot_training_history(st.session_state['history'])
            st.plotly_chart(fig, use_container_width=True)

elif page == "3. Feature Extraction":
    st.header("Step 3: Feature Extraction")
    
    st.markdown("""
    According to the paper:
    > "To generate high dimensional features of the ECG signals, we discarded fully-connected 
    > layers of trained CNN structure and kept the convolution layers which contain information 
    > of ECG signal."
    
    This step creates a feature extractor by removing the fully-connected layers and keeping 
    only the convolutional layers.
    """)
    
    if 'cnn' not in st.session_state:
        st.warning("Please train the model in Step 2 first!")
    else:
        if st.button("Create Feature Extractor"):
            with st.spinner("Creating feature extractor..."):
                cnn = st.session_state['cnn']
                feature_extractor = cnn.create_feature_extractor()
                
                st.session_state['feature_extractor'] = feature_extractor
                
                sample_features = cnn.extract_features(st.session_state['X_test'][:1])
                
                st.success("✓ Feature extractor created!")
                st.write(f"- **Feature vector dimension**: {sample_features.shape[1]}")
                st.write(f"- **Convolutional layers preserved**: {len([l for l in feature_extractor.layers if 'conv' in l.name])}")
        
        if 'feature_extractor' in st.session_state:
            st.subheader("Feature Extractor Architecture")
            
            cnn = st.session_state['cnn']
            
            st.write("**Layers in Feature Extractor:**")
            for layer in cnn.feature_extractor.layers:
                st.write(f"- {layer.__class__.__name__}: {layer.name}")
            
            st.write("\n**Layers Discarded (FC layers):**")
            include = False
            for layer in cnn.model.layers:
                if isinstance(layer, tf.keras.layers.Flatten):
                    include = True
                    continue
                if include and isinstance(layer, tf.keras.layers.Dense):
                    st.write(f"- {layer.__class__.__name__}: {layer.name} (output: {layer.units})")

elif page == "4. Authentication Testing":
    st.header("Step 4: Authentication Testing")
    
    st.markdown("""
    **User/Intruder Evaluation Protocol** (as per paper):
    > "We randomly divided the dataset into intruders and users for login test."
    
    - **Enrolled Users**: Subjects whose features are stored in the database
    - **Intruders**: Subjects attempting to authenticate (may be genuine users or imposters)
    - **Authentication**: Uses Euclidean distance between feature vectors
    - **Decision**: Accept if distance < threshold, otherwise reject
    """)
    
    if 'feature_extractor' not in st.session_state:
        st.warning("Please create the feature extractor in Step 3 first!")
    else:
        user_ratio = st.slider("Enrolled User Ratio", min_value=0.1, max_value=0.9, value=0.5, step=0.1)
        
        if st.button("Evaluate Authentication System"):
            with st.spinner("Evaluating authentication..."):
                cnn = st.session_state['cnn']
                subjects = st.session_state['subjects']
                
                authenticator = ECGAuthenticator(cnn)
                
                user_indices, intruder_indices = authenticator.split_users_intruders(
                    subjects, user_ratio=user_ratio, random_state=42
                )
                
                X_train = st.session_state['X_train']
                y_train = st.session_state['y_train']
                X_test = st.session_state['X_test']
                y_test = st.session_state['y_test']
                
                user_mask = np.isin(y_train, user_indices)
                X_enroll = X_train[user_mask]
                y_enroll = y_train[user_mask]
                
                authenticator.enroll_users(X_enroll, y_enroll, subjects)
                
                results = authenticator.evaluate_authentication(
                    X_test, y_test, subjects, user_indices, intruder_indices
                )
                
                st.session_state['auth_results'] = results
                st.session_state['authenticator'] = authenticator
                
                st.success("✓ Authentication evaluation complete!")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{results['accuracy']*100:.2f}%")
                col2.metric("FAR (False Accept)", f"{results['far']*100:.2f}%")
                col3.metric("FRR (False Reject)", f"{results['frr']*100:.2f}%")
                col4.metric("EER", f"{results['eer']*100:.2f}%")
                
                st.subheader("ROC Curve")
                fig_roc = plot_roc_curve(results['fpr'], results['tpr'], results['roc_auc'])
                st.plotly_chart(fig_roc, use_container_width=True)
                
                st.info(f"**Optimal Threshold**: {results['threshold']:.4f}")
                st.write(f"**Enrolled Users**: {len(user_indices)} subjects")
                st.write(f"**Test Pool**: {len(np.unique(y_test))} subjects")

elif page == "5. Optimization & Results":
    st.header("Step 5: Complexity Optimization & Final Results")
    
    st.markdown("""
    **Low Complexity Improvements** (as per paper):
    
    1. **Binary Weights**: Apply SIGN function to convert weights to {+1, -1}
       - Replaces multiplications with simple inversions
       
    2. **Approximate Weights**: Use exponential representation (w = 2^a + 2^b + ...)
       - Replaces multiplications with bit shifts and additions
       - Parameter n controls approximation accuracy
    
    **CPU Cycle Calculation** (AMD K7 architecture):
    - Multiplication: 3 cycles
    - Inversion: 1 cycle
    - Bit Shift: 1 cycle
    - Addition: 1 cycle
    """)
    
    if 'cnn' not in st.session_state:
        st.warning("Please complete Steps 1-4 first!")
    else:
        if st.button("Evaluate All Optimization Variants"):
            with st.spinner("Evaluating optimization variants... This may take a few minutes."):
                cnn = st.session_state['cnn']
                X_test = st.session_state['X_test']
                y_test = st.session_state['y_test']
                
                optimizer = WeightOptimizer(cnn.model)
                
                results = optimizer.evaluate_all_variants(X_test, y_test)
                
                st.session_state['optimization_results'] = results
                
                st.success("✓ Optimization evaluation complete!")
        
        if 'optimization_results' in st.session_state:
            results = st.session_state['optimization_results']
            
            st.subheader("Accuracy Comparison")
            
            df_acc = pd.DataFrame({
                'Variant': ['Original', 'Binary', 'Approx (n=1)', 'Approx (n=2)', 'Approx (n=3)'],
                'Accuracy (%)': [
                    results['original']['accuracy'] * 100,
                    results['binary']['accuracy'] * 100,
                    results['approx_n1']['accuracy'] * 100,
                    results['approx_n2']['accuracy'] * 100,
                    results['approx_n3']['accuracy'] * 100
                ],
                'CPU Cycles': [
                    results['original']['cpu_cycles'],
                    results['binary']['cpu_cycles'],
                    results['approx_n1']['cpu_cycles'],
                    results['approx_n2']['cpu_cycles'],
                    results['approx_n3']['cpu_cycles']
                ]
            })
            
            st.dataframe(df_acc, use_container_width=True)
            
            st.subheader("Paper Results Comparison")
            st.markdown("""
            **Expected Results from Paper**:
            - Original Weight: **99.63%** accuracy
            - Binary Weight: ~**91.06%** accuracy
            - Approximate Weight: ~**98.88%** accuracy (optimized)
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Accuracy vs. Complexity Trade-off")
                fig_complexity = plot_complexity_comparison(results)
                st.plotly_chart(fig_complexity, use_container_width=True)
            
            with col2:
                st.subheader("CPU Cycles Comparison")
                fig_cpu = go.Figure(data=[
                    go.Bar(
                        x=['Original', 'Binary', 'Approx\n(n=1)', 'Approx\n(n=2)', 'Approx\n(n=3)'],
                        y=[r['cpu_cycles'] for r in results.values()],
                        marker_color=['blue', 'green', 'orange', 'red', 'purple']
                    )
                ])
                fig_cpu.update_layout(
                    title='CPU Cycles by Weight Variant',
                    yaxis_title='CPU Cycles',
                    height=400
                )
                st.plotly_chart(fig_cpu, use_container_width=True)
            
            st.subheader("Operations Breakdown")
            fig_ops = plot_operations_comparison(results)
            st.plotly_chart(fig_ops, use_container_width=True)
            
            st.subheader("Detailed Operations Table")
            ops_data = []
            for variant, data in results.items():
                ops = data['operations']
                ops_data.append({
                    'Variant': variant,
                    'Multiplication': ops['multiplication'],
                    'Inversion': ops['inversion'],
                    'Bit Shift': ops['bit_shift'],
                    'Addition': ops['addition'],
                    'Total CPU Cycles': data['cpu_cycles']
                })
            
            df_ops = pd.DataFrame(ops_data)
            st.dataframe(df_ops, use_container_width=True)

elif page == "6. Live ECG Authentication":
    st.header("Step 6: Live ECG Signal Authentication")
    
    st.markdown("""
    Upload your own ECG signal for real-time authentication testing.
    
    **Requirements:**
    - Signal format: CSV, TXT, or numpy NPY file
    - Expected: Single-channel ECG signal  
    - The system will automatically:
      - Resample to 200 Hz
      - Detect R-peaks
      - Extract QRS complexes
      - Perform authentication against enrolled users
    """)
    
    if 'cnn' not in st.session_state or 'authenticator' not in st.session_state:
        st.warning("Please complete Steps 1-4 first to train the model and set up authentication!")
    else:
        uploaded_file = st.file_uploader("Upload ECG Signal", type=['csv', 'txt', 'npy'])
        
        if uploaded_file is not None:
            try:
                file_ext = uploaded_file.name.lower()
                
                if file_ext.endswith('.npy'):
                    ecg_data = np.load(uploaded_file)
                elif file_ext.endswith('.csv'):
                    ecg_data = np.loadtxt(uploaded_file, delimiter=',')
                else:
                    try:
                        ecg_data = np.loadtxt(uploaded_file)
                    except (ValueError, UnicodeDecodeError):
                        uploaded_file.seek(0)
                        ecg_data = np.loadtxt(uploaded_file, delimiter=',')
                
                if len(ecg_data.shape) > 1:
                    ecg_data = ecg_data.flatten()
                
                st.success(f"✓ Signal loaded: {len(ecg_data)} samples")
                
                col1, col2 = st.columns(2)
                sampling_rate = col1.number_input("Original Sampling Rate (Hz)", value=1000, min_value=100, max_value=5000)
                
                if col2.button("Process & Authenticate"):
                    with st.spinner("Processing ECG signal..."):
                        preprocessor = PTBDataPreprocessor(target_fs=200, window_size=1.0, num_complexes=20)
                        
                        resampled = preprocessor.resample_signal(ecg_data, sampling_rate)
                        
                        r_peaks = preprocessor.detect_r_peaks(resampled, 200)
                        
                        if len(r_peaks) == 0:
                            st.error("No R-peaks detected! Please check your ECG signal quality.")
                        else:
                            st.success(f"✓ Detected {len(r_peaks)} R-peaks")
                            
                            complexes = preprocessor.extract_qrs_complexes(resampled, r_peaks, 200)
                            
                            if len(complexes) == 0:
                                st.error("Could not extract QRS complexes!")
                            else:
                                st.info(f"Extracted {len(complexes)} QRS complexes")
                                
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(y=resampled, mode='lines', name='ECG Signal'))
                                fig.add_trace(go.Scatter(x=r_peaks, y=resampled[r_peaks], 
                                                       mode='markers', name='R-peaks',
                                                       marker=dict(color='red', size=10)))
                                fig.update_layout(title='ECG Signal with Detected R-peaks',
                                                xaxis_title='Sample Index',
                                                yaxis_title='Amplitude',
                                                height=400)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                num_to_test = min(len(complexes), 5)
                                test_complexes = complexes[:num_to_test].reshape(num_to_test, 200, 1)
                                
                                cnn = st.session_state['cnn']
                                authenticator = st.session_state['authenticator']
                                
                                features = cnn.extract_features(test_complexes)
                                
                                st.subheader("Authentication Results")
                                
                                auth_results = []
                                for idx, feature in enumerate(features):
                                    is_auth, match, distance = authenticator.authenticate(feature)
                                    auth_results.append({
                                        'Complex #': idx + 1,
                                        'Authenticated': '✓ Yes' if is_auth else '✗ No',
                                        'Best Match': match if match else 'None',
                                        'Distance': f"{distance:.4f}",
                                        'Threshold': f"{authenticator.threshold:.4f}"
                                    })
                                
                                df_results = pd.DataFrame(auth_results)
                                st.dataframe(df_results, use_container_width=True)
                                
                                auth_count = sum([1 for r in auth_results if r['Authenticated'] == '✓ Yes'])
                                st.metric("Authentication Rate", f"{auth_count}/{num_to_test} complexes passed")
                                
                                if auth_count > num_to_test / 2:
                                    st.success("🟢 User authenticated! Majority of complexes passed.")
                                else:
                                    st.error("🔴 Authentication failed! Majority of complexes rejected.")
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
        
        st.markdown("---")
        st.markdown("**Sample ECG Data Sources:**")
        st.markdown("- [PhysioNet](https://physionet.org/)")
        st.markdown("- Generate test data from the trained model's subjects")

elif page == "7. Advanced Optimizations":
    st.header("Step 7: Advanced Optimization Techniques")
    
    st.markdown("""
    Beyond binary and approximate weights, this section implements:
    - **TensorFlow Lite Quantization**: INT8 and FLOAT16 quantization
    - **Model Pruning**: Magnitude-based weight pruning with various sparsity levels
    """)
    
    if 'cnn' not in st.session_state:
        st.warning("Please train the model in Step 2 first!")
    else:
        st.subheader("TensorFlow Lite Quantization")
        
        st.markdown("""
        **Post-Training Quantization** reduces model size and improves inference speed:
        - **INT8**: 4x size reduction, fastest inference
        - **FLOAT16**: 2x size reduction, good balance
        """)
        
        if st.button("Apply TFLite Quantization"):
            with st.spinner("Applying quantization..."):
                import tempfile
                
                cnn = st.session_state['cnn']
                X_test = st.session_state['X_test']
                y_test = st.session_state['y_test']
                
                with tempfile.TemporaryDirectory() as tmpdir:
                    model_path = os.path.join(tmpdir, 'model.keras')
                    cnn.model.save(model_path)
                    
                    converter = tf.lite.TFLiteConverter.from_keras_model(cnn.model)
                    
                    tflite_model = converter.convert()
                    
                    converter_float16 = tf.lite.TFLiteConverter.from_keras_model(cnn.model)
                    converter_float16.optimizations = [tf.lite.Optimize.DEFAULT]
                    converter_float16.target_spec.supported_types = [tf.float16]
                    tflite_float16 = converter_float16.convert()
                    
                    def representative_dataset():
                        for i in range(100):
                            yield [X_test[i:i+1].astype(np.float32)]
                    
                    converter_int8 = tf.lite.TFLiteConverter.from_keras_model(cnn.model)
                    converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
                    converter_int8.representative_dataset = representative_dataset
                    converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                    converter_int8.inference_input_type = tf.int8
                    converter_int8.inference_output_type = tf.int8
                    
                    try:
                        tflite_int8 = converter_int8.convert()
                        int8_available = True
                    except Exception as e:
                        st.warning(f"INT8 quantization not fully supported: {e}")
                        int8_available = False
                    
                    original_size = len(tf.io.serialize_tensor(tf.constant(cnn.model.get_weights()[0]))) * len(cnn.model.get_weights())
                    
                    quant_results = {
                        'Original': {
                            'size': original_size,
                            'compression': 1.0
                        },
                        'FLOAT16': {
                            'size': len(tflite_float16),
                            'compression': original_size / len(tflite_float16)
                        }
                    }
                    
                    if int8_available:
                        quant_results['INT8'] = {
                            'size': len(tflite_int8),
                            'compression': original_size / len(tflite_int8)
                        }
                    
                    st.session_state['quant_results'] = quant_results
                    st.session_state['tflite_models'] = {
                        'float32': tflite_model,
                        'float16': tflite_float16
                    }
                    if int8_available:
                        st.session_state['tflite_models']['int8'] = tflite_int8
                    
                    st.success("✓ Quantization complete!")
        
        if 'quant_results' in st.session_state:
            st.subheader("Quantization Results")
            
            quant_data = []
            for variant, data in st.session_state['quant_results'].items():
                quant_data.append({
                    'Variant': variant,
                    'Size (bytes)': f"{data['size']:,}",
                    'Compression Ratio': f"{data['compression']:.2f}x"
                })
            
            df_quant = pd.DataFrame(quant_data)
            st.dataframe(df_quant, use_container_width=True)
            
            fig = go.Figure(data=[
                go.Bar(x=[d['Variant'] for d in quant_data],
                      y=[st.session_state['quant_results'][d['Variant']]['size'] for d in quant_data],
                      marker_color=['blue', 'green', 'orange'])
            ])
            fig.update_layout(title='Model Size Comparison',
                            yaxis_title='Size (bytes)',
                            height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Manual Weight Pruning")
        
        st.markdown("""
        **Magnitude-based Pruning** removes small weights to create sparse models:
        - Reduces model complexity
        - Can improve inference speed on specialized hardware
        - May reduce accuracy slightly
        
        This implementation uses manual thresholding to prune weights.
        """)
        
        sparsity = st.slider("Pruning Sparsity (%)", min_value=0, max_value=90, value=50, step=10)
        
        if st.button("Apply Pruning"):
            with st.spinner(f"Pruning model to {sparsity}% sparsity..."):
                cnn = st.session_state['cnn']
                X_test = st.session_state['X_test']
                y_test = st.session_state['y_test']
                
                pruned_model = keras.models.clone_model(cnn.model)
                pruned_model.set_weights(cnn.model.get_weights())
                
                threshold_percentile = sparsity
                
                for layer in pruned_model.layers:
                    if isinstance(layer, (keras.layers.Conv1D, keras.layers.Dense)):
                        weights = layer.get_weights()
                        if len(weights) > 0:
                            kernel = weights[0]
                            threshold = np.percentile(np.abs(kernel), threshold_percentile)
                            pruned_kernel = kernel.copy()
                            pruned_kernel[np.abs(kernel) < threshold] = 0
                            weights[0] = pruned_kernel
                            layer.set_weights(weights)
                
                pruned_model.compile(
                    optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                _, pruned_acc = pruned_model.evaluate(X_test, y_test, verbose=0)
                
                total_weights = 0
                zero_weights = 0
                for layer in pruned_model.layers:
                    if isinstance(layer, (keras.layers.Conv1D, keras.layers.Dense)):
                        weights = layer.get_weights()
                        if len(weights) > 0:
                            kernel = weights[0]
                            total_weights += kernel.size
                            zero_weights += np.sum(kernel == 0)
                
                actual_sparsity = (zero_weights / total_weights * 100) if total_weights > 0 else 0
                
                st.session_state['pruned_model'] = pruned_model
                st.session_state['pruned_accuracy'] = pruned_acc
                st.session_state['pruning_sparsity'] = actual_sparsity
                
                st.success(f"✓ Pruning complete!")
                st.write(f"**Pruned Model Accuracy**: {pruned_acc*100:.2f}%")
                st.write(f"**Actual Sparsity Level**: {actual_sparsity:.2f}%")
                st.write(f"**Zero Weights**: {zero_weights:,} / {total_weights:,}")

elif page == "8. Cross-Validation":
    st.header("Step 8: Cross-Validation & Robustness Testing")
    
    st.markdown("""
    Evaluate authentication robustness with different user/intruder splits:
    - K-fold cross-validation
    - Multiple random splits
    - Statistical analysis of performance variance
    """)
    
    if 'cnn' not in st.session_state:
        st.warning("Please complete Steps 1-4 first!")
    else:
        num_splits = st.number_input("Number of Cross-Validation Splits", min_value=3, max_value=10, value=5)
        user_ratio = st.slider("User Ratio", min_value=0.3, max_value=0.7, value=0.5, step=0.1)
        
        if st.button("Run Cross-Validation"):
            with st.spinner("Running cross-validation..."):
                cnn = st.session_state['cnn']
                subjects = st.session_state['subjects']
                X_test = st.session_state['X_test']
                y_test = st.session_state['y_test']
                X_train = st.session_state['X_train']
                y_train = st.session_state['y_train']
                
                cv_results = []
                
                for split_idx in range(num_splits):
                    authenticator = ECGAuthenticator(cnn)
                    
                    user_indices, intruder_indices = authenticator.split_users_intruders(
                        subjects, user_ratio=user_ratio, random_state=split_idx*42
                    )
                    
                    user_mask = np.isin(y_train, user_indices)
                    X_enroll = X_train[user_mask]
                    y_enroll = y_train[user_mask]
                    
                    authenticator.enroll_users(X_enroll, y_enroll, subjects)
                    
                    results = authenticator.evaluate_authentication(
                        X_test, y_test, subjects, user_indices, intruder_indices
                    )
                    
                    cv_results.append({
                        'Split': split_idx + 1,
                        'Accuracy': results['accuracy'],
                        'FAR': results['far'],
                        'FRR': results['frr'],
                        'EER': results['eer'],
                        'AUC': results['roc_auc']
                    })
                
                st.session_state['cv_results'] = cv_results
                
                df_cv = pd.DataFrame(cv_results)
                
                st.success("✓ Cross-validation complete!")
                
                st.subheader("Cross-Validation Results")
                st.dataframe(df_cv, use_container_width=True)
                
                st.subheader("Statistical Summary")
                col1, col2, col3 = st.columns(3)
                
                accuracies = [r['Accuracy'] for r in cv_results]
                col1.metric("Mean Accuracy", f"{np.mean(accuracies)*100:.2f}%")
                col1.metric("Std Dev", f"{np.std(accuracies)*100:.2f}%")
                
                fars = [r['FAR'] for r in cv_results]
                col2.metric("Mean FAR", f"{np.mean(fars)*100:.2f}%")
                col2.metric("Std Dev", f"{np.std(fars)*100:.2f}%")
                
                frrs = [r['FRR'] for r in cv_results]
                col3.metric("Mean FRR", f"{np.mean(frrs)*100:.2f}%")
                col3.metric("Std Dev", f"{np.std(frrs)*100:.2f}%")
                
                st.subheader("Performance Across Splits")
                fig = go.Figure()
                fig.add_trace(go.Box(y=accuracies, name='Accuracy'))
                fig.add_trace(go.Box(y=fars, name='FAR'))
                fig.add_trace(go.Box(y=frrs, name='FRR'))
                fig.update_layout(title='Performance Metrics Distribution',
                                yaxis_title='Value',
                                height=400)
                st.plotly_chart(fig, use_container_width=True)

elif page == "9. Model Export":
    st.header("Step 9: Model Export for IoT Deployment")
    
    st.markdown("""
    Export trained models for deployment on IoT edge devices:
    - **TensorFlow Lite**: Optimized for mobile/embedded devices
    - **ONNX**: Cross-platform ML model format
    - **C++ Header**: For ARM Cortex-M microcontrollers
    """)
    
    if 'cnn' not in st.session_state:
        st.warning("Please train the model in Step 2 first!")
    else:
        st.subheader("Export Options")
        
        export_format = st.selectbox("Select Export Format", [
            "TensorFlow Lite (.tflite)",
            "ONNX (.onnx)",
            "Keras (.keras)",
            "C++ Header (.h)"
        ])
        
        col1, col2 = st.columns(2)
        include_fc = col1.checkbox("Include Fully-Connected Layers", value=False,
                                   help="Check to export full classifier, uncheck for feature extractor only")
        quantize = col2.checkbox("Apply Quantization (TFLite only)", value=False)
        
        if st.button("Export Model"):
            with st.spinner(f"Exporting to {export_format}..."):
                cnn = st.session_state['cnn']
                model_to_export = cnn.model if include_fc else cnn.feature_extractor
                
                if export_format == "TensorFlow Lite (.tflite)":
                    converter = tf.lite.TFLiteConverter.from_keras_model(model_to_export)
                    
                    if quantize:
                        converter.optimizations = [tf.lite.Optimize.DEFAULT]
                        X_test = st.session_state['X_test']
                        
                        def representative_dataset():
                            for i in range(100):
                                yield [X_test[i:i+1].astype(np.float32)]
                        
                        converter.representative_dataset = representative_dataset
                    
                    tflite_model = converter.convert()
                    
                    filename = "ecg_auth_model.tflite"
                    with open(filename, 'wb') as f:
                        f.write(tflite_model)
                    
                    st.success(f"✓ Model exported to {filename}")
                    st.download_button(
                        label="Download TFLite Model",
                        data=tflite_model,
                        file_name=filename,
                        mime="application/octet-stream"
                    )
                    
                    st.info(f"Model size: {len(tflite_model):,} bytes ({len(tflite_model)/1024:.2f} KB)")
                
                elif export_format == "ONNX (.onnx)":
                    try:
                        import tf2onnx
                        
                        spec = (tf.TensorSpec((None, 200, 1), tf.float32, name="input"),)
                        
                        import tempfile
                        with tempfile.TemporaryDirectory() as tmpdir:
                            model_path = os.path.join(tmpdir, 'temp_model')
                            model_to_export.save(model_path)
                            
                            onnx_model, _ = tf2onnx.convert.from_keras(
                                model_to_export,
                                input_signature=spec,
                                opset=13
                            )
                            
                            filename = "ecg_auth_model.onnx"
                            with open(filename, 'wb') as f:
                                f.write(onnx_model.SerializeToString())
                            
                            st.success(f"✓ Model exported to {filename}")
                            
                            with open(filename, 'rb') as f:
                                st.download_button(
                                    label="Download ONNX Model",
                                    data=f,
                                    file_name=filename,
                                    mime="application/octet-stream"
                                )
                    except ImportError:
                        st.error("tf2onnx not installed. Install with: pip install tf2onnx")
                
                elif export_format == "Keras (.keras)":
                    filename = "ecg_auth_model.keras"
                    model_to_export.save(filename)
                    
                    st.success(f"✓ Model exported to {filename}")
                    
                    with open(filename, 'rb') as f:
                        st.download_button(
                            label="Download Keras Model",
                            data=f,
                            file_name=filename,
                            mime="application/octet-stream"
                        )
                
                elif export_format == "C++ Header (.h)":
                    converter = tf.lite.TFLiteConverter.from_keras_model(model_to_export)
                    if quantize:
                        converter.optimizations = [tf.lite.Optimize.DEFAULT]
                    
                    tflite_model = converter.convert()
                    
                    model_bytes = [f"0x{b:02x}" for b in tflite_model]
                    
                    header_content = f"""#ifndef ECG_AUTH_MODEL_H
#define ECG_AUTH_MODEL_H

// Auto-generated TensorFlow Lite model for ECG authentication
// Model type: {'Full Classifier' if include_fc else 'Feature Extractor'}
// Quantized: {'Yes' if quantize else 'No'}
// Size: {len(tflite_model)} bytes

const unsigned char ecg_auth_model[] = {{
  {', '.join(model_bytes[:16])}{',' if len(model_bytes) > 16 else ''}
"""
                    
                    for i in range(16, len(model_bytes), 16):
                        chunk = model_bytes[i:i+16]
                        header_content += f"  {', '.join(chunk)}{',' if i + 16 < len(model_bytes) else ''}\n"
                    
                    header_content += f"""
}};

const unsigned int ecg_auth_model_len = {len(tflite_model)};

#endif // ECG_AUTH_MODEL_H
"""
                    
                    filename = "ecg_auth_model.h"
                    with open(filename, 'w') as f:
                        f.write(header_content)
                    
                    st.success(f"✓ Model exported to {filename}")
                    st.download_button(
                        label="Download C++ Header",
                        data=header_content,
                        file_name=filename,
                        mime="text/plain"
                    )
                    
                    st.code(header_content[:500] + "...", language="cpp")
        
        st.markdown("---")
        st.markdown("""
        **Deployment Instructions:**
        
        **TensorFlow Lite (Mobile/Edge):**
        ```python
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path="ecg_auth_model.tflite")
        interpreter.allocate_tensors()
        ```
        
        **ARM Cortex-M (Microcontrollers):**
        1. Include the generated .h file in your project
        2. Use TensorFlow Lite Micro runtime
        3. Example:
        ```cpp
        #include "ecg_auth_model.h"
        #include "tensorflow/lite/micro/all_ops_resolver.h"
        #include "tensorflow/lite/micro/micro_interpreter.h"
        ```
        """)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### Paper Reference
Wang, G., John, D., & Nag, A. (2020). 
*Low Complexity ECG Biometric Authentication for IoT Edge Devices*.
IEEE ICTA 2020.
""")

st.sidebar.markdown("""
### Implementation Details
- **Database**: PTB Diagnostic ECG (290 subjects)
- **Sampling Rate**: 200 Hz
- **Window Size**: 1 second (200 samples)
- **Complexes/Subject**: 20
- **CNN Architecture**: 3 Conv1D layers
- **Training**: 20 epochs, batch size 16
""")
