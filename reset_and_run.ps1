# Reset script for ai_market_predictor

Write-Host ">>> Stopping any Python processes..."
taskkill /F /IM python.exe /T 2>$null

Write-Host ">>> Deleting old artifacts..."
if (Test-Path artifacts) {
    Remove-Item -Recurse -Force artifacts
}

Write-Host ">>> Clearing Streamlit cache..."
if (Test-Path "$env:USERPROFILE\.streamlit") {
    Remove-Item -Recurse -Force "$env:USERPROFILE\.streamlit"
}
if (Test-Path "$env:TEMP\streamlit") {
    Remove-Item -Recurse -Force "$env:TEMP\streamlit"
}

Write-Host ">>> Activating virtual environment..."
& .\.venv\Scripts\Activate.ps1

Write-Host ">>> Starting Streamlit app..."
python -m streamlit run streamlit_app.py --server.port=8501
