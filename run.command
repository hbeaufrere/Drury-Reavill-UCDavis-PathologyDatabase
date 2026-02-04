#!/bin/bash
# Double-click this file in Finder to launch the Drury Dataset Explorer
cd "$(dirname "$0")"
echo "Starting Drury Dataset Explorer..."
echo "The app will open in your browser at http://localhost:8501"
echo ""
/Users/huguesbeaufrere/Library/Python/3.9/bin/streamlit run app.py
