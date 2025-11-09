import numpy as np
import tensorflow as tf
from tensorflow import keras
import copy

class WeightOptimizer:
    def __init__(self, model):
        self.original_model = model
        self.optimized_models = {}
        
    def binary_weight_transform(self, weights):
        """Apply SIGN function to convert weights to binary (+1, -1)"""
        binary_weights = np.sign(weights)
        binary_weights[binary_weights == 0] = 1
        return binary_weights
    
    def approximate_weight_transform(self, weights, n=1):
        """Approximate weights using exponential representation"""
        approx_weights = np.zeros_like(weights)
        
        for i in range(len(weights.flat)):
            w = weights.flat[i]
            
            if w == 0:
                approx_weights.flat[i] = 0
                continue
            
            sign = np.sign(w)
            abs_w = np.abs(w)
            
            if abs_w < 1e-10:
                approx_weights.flat[i] = 0
                continue
            
            log_w = np.log2(abs_w)
            base_exp = int(np.floor(log_w))
            
            approx_sum = 2 ** base_exp
            
            for j in range(1, n):
                remainder = abs_w - approx_sum
                if remainder <= 0:
                    break
                log_remainder = np.log2(remainder) if remainder > 0 else base_exp - 10
                next_exp = int(np.floor(log_remainder))
                approx_sum += 2 ** next_exp
            
            approx_weights.flat[i] = sign * approx_sum
        
        return approx_weights
    
    def create_binary_model(self):
        """Create model with binary weights"""
        binary_model = keras.models.clone_model(self.original_model)
        binary_model.set_weights(self.original_model.get_weights())
        
        for layer in binary_model.layers:
            if isinstance(layer, (keras.layers.Conv1D, keras.layers.Dense)):
                weights = layer.get_weights()
                if len(weights) > 0:
                    binary_kernel = self.binary_weight_transform(weights[0])
                    weights[0] = binary_kernel
                    layer.set_weights(weights)
        
        self.optimized_models['binary'] = binary_model
        return binary_model
    
    def create_approximate_model(self, n=1):
        """Create model with approximate weights"""
        approx_model = keras.models.clone_model(self.original_model)
        approx_model.set_weights(self.original_model.get_weights())
        
        for layer in approx_model.layers:
            if isinstance(layer, (keras.layers.Conv1D, keras.layers.Dense)):
                weights = layer.get_weights()
                if len(weights) > 0:
                    approx_kernel = self.approximate_weight_transform(weights[0], n=n)
                    weights[0] = approx_kernel
                    layer.set_weights(weights)
        
        self.optimized_models[f'approx_n{n}'] = approx_model
        return approx_model
    
    def count_operations(self, model):
        """Count operations for complexity analysis"""
        total_mult = total_inv = total_shift = total_add = 0

        for layer in model.layers:
            if isinstance(layer, keras.layers.Conv1D):
                weights = layer.get_weights()
                if len(weights) > 0:
                    kernel = weights[0]
                    try:
                        shape = getattr(layer, "output_shape", None)
                        if shape is None or not hasattr(shape, "__len__"):
                            shape = getattr(layer.output, "shape", None)
                        if shape is not None and len(shape) > 1:
                            output_length = shape[1] if shape[1] is not None else 100
                            num_filters = shape[2] if len(shape) > 2 else kernel.shape[-1]
                        else:
                            output_length, num_filters = 100, kernel.shape[-1]
                    except Exception:
                        output_length, num_filters = 100, kernel.shape[-1]

                    kernel_size, input_channels = kernel.shape[0], kernel.shape[1]
                    ops_per_output = kernel_size * input_channels
                    total_ops = ops_per_output * num_filters * output_length

                    if np.all(np.abs(kernel) == 1):
                        total_inv += total_ops
                    elif self._is_power_of_2_weights(kernel):
                        n = self._estimate_n_from_weights(kernel)
                        total_shift += total_ops * n
                        total_add += total_ops * n
                        total_inv += int(total_ops * 0.2)
                    else:
                        total_mult += total_ops

            elif isinstance(layer, keras.layers.Dense):
                weights = layer.get_weights()
                if len(weights) > 0:
                    kernel = weights[0]
                    total_ops = kernel.size
                    if np.all(np.abs(kernel) == 1):
                        total_inv += total_ops
                    elif self._is_power_of_2_weights(kernel):
                        n = self._estimate_n_from_weights(kernel)
                        total_shift += total_ops * n
                        total_add += total_ops * n
                        total_inv += int(total_ops * 0.2)
                    else:
                        total_mult += total_ops

        # safer bias addition calculation
        bias_adds = 0
        for layer in model.layers:
            if isinstance(layer, (keras.layers.Conv1D, keras.layers.Dense)) and getattr(layer, "use_bias", False):
                try:
                    shape = getattr(layer, "output_shape", None)
                    if shape is None or not hasattr(shape, "__len__"):
                        shape = getattr(layer.output, "shape", None)
                    if shape is not None:
                        bias_adds += np.prod(shape[1:])
                except Exception:
                    continue
        total_add += int(bias_adds) if bias_adds else 0

        return {
            "multiplication": total_mult,
            "inversion": total_inv,
            "bit_shift": total_shift,
            "addition": total_add,
        }
    
    def _is_power_of_2_weights(self, weights):
        """Check if weights are power of 2 based"""
        non_zero = weights[weights != 0]
        if len(non_zero) == 0:
            return False
        log2_vals = np.log2(np.abs(non_zero))
        return np.allclose(log2_vals, log2_vals.astype(int), atol=0.01)
    
    def _estimate_n_from_weights(self, weights):
        """Estimate n parameter from approximate weights"""
        return 1
    
    def calculate_cpu_cycles(self, operations):
        """Calculate CPU cycles based on AMD K7 architecture"""
        cycles = (
            operations['multiplication'] * 3 +
            operations['inversion'] * 1 +
            operations['bit_shift'] * 1 +
            operations['addition'] * 1
        )
        return cycles
    
    def evaluate_all_variants(self, X_test, y_test):
        """Evaluate all weight variants"""
        results = {}
        
        print("Evaluating original model...")
        original_pred = self.original_model.predict(X_test, verbose=0)
        original_acc = np.mean(np.argmax(original_pred, axis=1) == y_test)
        original_ops = self.count_operations(self.original_model)
        original_cycles = self.calculate_cpu_cycles(original_ops)
        
        results['original'] = {
            'accuracy': original_acc,
            'operations': original_ops,
            'cpu_cycles': original_cycles
        }
        
        print("Creating and evaluating binary model...")
        binary_model = self.create_binary_model()
        binary_pred = binary_model.predict(X_test, verbose=0)
        binary_acc = np.mean(np.argmax(binary_pred, axis=1) == y_test)
        binary_ops = self.count_operations(binary_model)
        binary_cycles = self.calculate_cpu_cycles(binary_ops)
        
        results['binary'] = {
            'accuracy': binary_acc,
            'operations': binary_ops,
            'cpu_cycles': binary_cycles
        }
        
        for n in [1, 2, 3]:
            print(f"Creating and evaluating approximate model (n={n})...")
            approx_model = self.create_approximate_model(n=n)
            approx_pred = approx_model.predict(X_test, verbose=0)
            approx_acc = np.mean(np.argmax(approx_pred, axis=1) == y_test)
            approx_ops = self.count_operations(approx_model)
            approx_cycles = self.calculate_cpu_cycles(approx_ops)
            
            results[f'approx_n{n}'] = {
                'accuracy': approx_acc,
                'operations': approx_ops,
                'cpu_cycles': approx_cycles
            }
        
        return results
