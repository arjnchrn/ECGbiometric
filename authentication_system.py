import numpy as np
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split

class ECGAuthenticator:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
        self.enrolled_users = {}
        self.threshold = None
        
    def enroll_users(self, X, y, subjects):
        """Enroll users into the authentication database"""
        features = self.feature_extractor.extract_features(X)
        
        for idx, subject_id in enumerate(np.unique(y)):
            subject_features = features[y == subject_id]
            subject_name = subjects[subject_id]
            
            self.enrolled_users[subject_name] = {
                'subject_id': subject_id,
                'features': subject_features
            }
        
        print(f"Enrolled {len(self.enrolled_users)} users")
    
    def calculate_euclidean_distance(self, feature1, feature2):
        """Calculate Euclidean distance between two feature vectors"""
        return np.sqrt(np.sum((feature1 - feature2) ** 2))
    
    def authenticate(self, test_features, threshold=None):
        """Authenticate using Euclidean distance matching"""
        if threshold is None:
            threshold = self.threshold
        
        if threshold is None:
            raise ValueError("Threshold must be set!")
        
        min_distance = float('inf')
        best_match = None
        
        for user_name, user_data in self.enrolled_users.items():
            enrolled_features = user_data['features']
            
            for enrolled_feature in enrolled_features:
                distance = self.calculate_euclidean_distance(test_features, enrolled_feature)
                
                if distance < min_distance:
                    min_distance = distance
                    best_match = user_name
        
        is_authenticated = min_distance < threshold
        
        return is_authenticated, best_match, min_distance
    
    def evaluate_authentication(self, X_test, y_test, subjects, user_indices, intruder_indices):
        """Evaluate authentication system with user/intruder protocol"""
        test_features = self.feature_extractor.extract_features(X_test)
        
        y_true = []
        y_scores = []
        
        for idx in range(len(y_test)):
            test_feature = test_features[idx]
            true_subject_id = y_test[idx]
            
            is_genuine = true_subject_id in user_indices
            
            min_distance = float('inf')
            for user_name, user_data in self.enrolled_users.items():
                enrolled_features = user_data['features']
                
                for enrolled_feature in enrolled_features:
                    distance = self.calculate_euclidean_distance(test_feature, enrolled_feature)
                    if distance < min_distance:
                        min_distance = distance
            
            y_true.append(1 if is_genuine else 0)
            y_scores.append(-min_distance)
        
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        eer_idx = np.argmin(np.abs(fpr - (1 - tpr)))
        eer = fpr[eer_idx]
        eer_threshold = -thresholds[eer_idx]
        
        self.threshold = eer_threshold
        
        y_pred = (y_scores >= -eer_threshold).astype(int)
        accuracy = accuracy_score(y_true, y_pred)
        
        far = np.sum((y_pred == 1) & (y_true == 0)) / np.sum(y_true == 0) if np.sum(y_true == 0) > 0 else 0
        frr = np.sum((y_pred == 0) & (y_true == 1)) / np.sum(y_true == 1) if np.sum(y_true == 1) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'far': far,
            'frr': frr,
            'eer': eer,
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
            'threshold': eer_threshold
        }
    
    def split_users_intruders(self, subjects, user_ratio=0.5, random_state=42):
        """Randomly split subjects into enrolled users and intruders"""
        np.random.seed(random_state)
        
        subject_ids = list(range(len(subjects)))
        np.random.shuffle(subject_ids)
        
        split_point = int(len(subject_ids) * user_ratio)
        user_indices = subject_ids[:split_point]
        intruder_indices = subject_ids[split_point:]
        
        return user_indices, intruder_indices
