import numpy as np
import wfdb
import os
from scipy import signal
from biosppy.signals import ecg
import pickle

class PTBDataPreprocessor:
    def __init__(self, target_fs=200, window_size=1.0, num_complexes=20):
        self.target_fs = target_fs
        self.window_size = window_size
        self.num_complexes = num_complexes
        self.data_dir = 'ptb_data'
        
    def download_ptb_database(self, num_subjects=None):
        """Download PTB database using wfdb"""
        print("Downloading PTB database...")
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        try:
            wfdb.dl_database('ptbdb', dl_dir=self.data_dir)
            print(f"Database downloaded to {self.data_dir}")
        except Exception as e:
            print(f"Download error (may already exist): {e}")
    
    def get_record_list(self):
        """Get list of all patient records from PTB database"""
        records = []
        
        if not os.path.exists(self.data_dir):
            return records
        
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.hea') and 'patient' in root:
                    record_path = os.path.join(root, file[:-4])
                    record_path = record_path.replace(self.data_dir + os.sep, '')
                    records.append(record_path)
        
        return sorted(records)
    
    def resample_signal(self, ecg_signal, original_fs):
        """Resample ECG signal to target frequency"""
        if original_fs == self.target_fs:
            return ecg_signal
        
        num_samples = int(len(ecg_signal) * self.target_fs / original_fs)
        resampled = signal.resample(ecg_signal, num_samples)
        return resampled
    
    def detect_r_peaks(self, ecg_signal, fs):
        """Detect R peaks in ECG signal using biosppy"""
        try:
            out = ecg.ecg(signal=ecg_signal, sampling_rate=fs, show=False)
            r_peaks = out['rpeaks']
            return r_peaks
        except Exception as e:
            print(f"R-peak detection error: {e}")
            return np.array([])
    
    def extract_qrs_complexes(self, ecg_signal, r_peaks, fs):
        """Extract QRS complexes around R peaks"""
        complexes = []
        half_window = int(self.window_size * fs / 2)
        
        for r_peak in r_peaks:
            start = r_peak - half_window
            end = r_peak + half_window
            
            if start >= 0 and end <= len(ecg_signal):
                complex_segment = ecg_signal[start:end]
                
                if len(complex_segment) == 2 * half_window:
                    complexes.append(complex_segment)
        
        return np.array(complexes)
    
    def process_single_record(self, record_path):
        """Process a single ECG record"""
        try:
            full_path = os.path.join(self.data_dir, record_path)
            record = wfdb.rdrecord(full_path)
            
            ecg_signal = record.p_signal[:, 0]
            original_fs = record.fs
            
            resampled_signal = self.resample_signal(ecg_signal, original_fs)
            
            r_peaks = self.detect_r_peaks(resampled_signal, self.target_fs)
            
            if len(r_peaks) == 0:
                return None
            
            complexes = self.extract_qrs_complexes(resampled_signal, r_peaks, self.target_fs)
            
            if len(complexes) < self.num_complexes:
                return None
            
            indices = np.random.choice(len(complexes), self.num_complexes, replace=False)
            selected_complexes = complexes[indices]
            
            return selected_complexes
            
        except Exception as e:
            print(f"Error processing {record_path}: {e}")
            return None
    
    def load_and_preprocess_data(self, max_subjects=290):
        """Load and preprocess PTB database"""
        print("Loading and preprocessing PTB database...")
        
        records = self.get_record_list()
        print(f"Found {len(records)} records")
        
        subject_data = {}
        
        for record in records:
            patient_id = record.split(os.sep)[0]
            
            if len(subject_data) >= max_subjects:
                break
            
            if patient_id not in subject_data:
                complexes = self.process_single_record(record)
                
                if complexes is not None:
                    subject_data[patient_id] = complexes
                    print(f"Processed {len(subject_data)}/{max_subjects}: {patient_id}")
        
        print(f"\nSuccessfully processed {len(subject_data)} subjects")
        
        subjects = list(subject_data.keys())
        X = []
        y = []
        
        for idx, subject in enumerate(subjects):
            complexes = subject_data[subject]
            for complex_signal in complexes:
                X.append(complex_signal)
                y.append(idx)
        
        X = np.array(X)
        y = np.array(y)
        
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        print(f"Final dataset shape: X={X.shape}, y={y.shape}")
        print(f"Number of subjects: {len(subjects)}")
        
        return X, y, subjects
    
    def save_processed_data(self, X, y, subjects, filename='processed_data.pkl'):
        """Save preprocessed data"""
        data = {
            'X': X,
            'y': y,
            'subjects': subjects,
            'params': {
                'target_fs': self.target_fs,
                'window_size': self.window_size,
                'num_complexes': self.num_complexes
            }
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Data saved to {filename}")
    
    def load_processed_data(self, filename='processed_data.pkl'):
        """Load preprocessed data"""
        if not os.path.exists(filename):
            return None, None, None
        
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        return data['X'], data['y'], data['subjects']
