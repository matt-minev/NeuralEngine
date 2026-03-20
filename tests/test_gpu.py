import sys
import os

# Add parent directory to sys.path to import the engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import neural_backend

print("Checking cupy module availability...")
try:
    import cupy
    print(f"CuPy version: {cupy.__version__}")
except ImportError:
    print("CuPy is NOT installed in this environment.")

print("\nChecking backend resolution (requesting GPU)...")
try:
    xp, device_name, backend_name, is_gpu = neural_backend.resolve_backend(device="gpu", warn=True)
    print(f"Device Name : {device_name}")
    print(f"Backend Name: {backend_name}")
    print(f"Is GPU      : {is_gpu}")
except Exception as e:
    print(f"Error resolving backend: {e}")

print("\nRunning a basic tensor allocation test...")
try:
    tensor = neural_backend.to_device([1.0, 2.0, 3.0], xp)
    print(f"Tensor type: {type(tensor)}")
    print("Test finished.")
except Exception as e:
    print(f"Error during allocation: {e}")
