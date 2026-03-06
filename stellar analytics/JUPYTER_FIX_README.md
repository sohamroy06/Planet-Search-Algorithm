# Jupyter Kernel Fix Guide

## Problem
You get `ModuleNotFoundError: No module named 'pandas'` even though you've installed the packages.

## Root Cause
Jupyter is using a different Python interpreter than where your packages are installed. This is common when you have multiple Python installations.

## Solution

### Quick Fix (Recommended)
1. **Run the fix script:**
   ```cmd
   fix_jupyter_kernel.bat
   ```

2. **Restart Jupyter** (if it's running)

3. **Change the kernel in your notebook:**
   
   **In Jupyter Notebook:**
   - Open `kepler_analysis.ipynb`
   - Click: `Kernel` → `Change kernel` → Select `Python (Kepler Analysis)`
   
   **In VS Code:**
   - Open `kepler_analysis.ipynb`
   - Click the kernel selector in the top-right corner
   - Select `Python (Kepler Analysis)` from the dropdown

4. **Run your cells again** - the imports should now work!

---

## Alternative Solutions

### Option 1: Install packages directly in the notebook
Add this as the first cell in your notebook and run it:
```python
import sys
!{sys.executable} -m pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

### Option 2: Use the correct pip command
Instead of just `pip install`, use:
```cmd
python -m pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter notebook ipykernel
```

This ensures packages are installed in the same Python environment that runs when you type `python`.

---

## Verification

After applying the fix, run this in a notebook cell to verify:
```python
import sys
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
print("\n✓ All packages imported successfully!")
```

---

## Still Having Issues?

If you're still having problems, check:
1. Which Python you're using: Run `where python` in cmd
2. Which kernels are available: Run `jupyter kernelspec list`
3. Verify packages are installed: Run `python -m pip list | findstr pandas`
