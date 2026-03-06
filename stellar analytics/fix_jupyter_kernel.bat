@echo off
echo ================================================================================
echo Fixing Jupyter Kernel for Kepler Analysis
echo ================================================================================
echo.

echo Step 1: Installing packages for current Python environment...
python -m pip install --upgrade pip
python -m pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter notebook ipykernel

echo.
echo Step 2: Registering Python kernel with Jupyter...
python -m ipykernel install --user --name=kepler_env --display-name="Python (Kepler Analysis)"

echo.
echo ================================================================================
echo Setup Complete!
echo ================================================================================
echo.
echo IMPORTANT: After running this script:
echo 1. Restart Jupyter if it's currently running
echo 2. Open your notebook: jupyter notebook kepler_analysis.ipynb
echo 3. Click Kernel ^> Change kernel ^> "Python (Kepler Analysis)"
echo.
echo Alternatively, in VS Code:
echo - Click on the kernel selector (top-right of notebook)
echo - Select "Python (Kepler Analysis)" from the list
echo.
pause
