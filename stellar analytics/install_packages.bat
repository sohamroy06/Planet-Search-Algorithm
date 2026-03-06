@echo off
echo ================================================================================
echo Installing Required Python Packages for Kepler Telescope Data Analysis
echo ================================================================================
echo.

echo Installing packages... This may take a few minutes.
echo.

pip install --upgrade pip
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter notebook ipykernel

echo.
echo ================================================================================
echo Installation Complete!
echo ================================================================================
echo.
echo You can now run the analysis using:
echo   python kepler_analysis.py
echo.
echo Or open the Jupyter notebook:
echo   jupyter notebook kepler_analysis.ipynb
echo.
pause
