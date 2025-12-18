#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from crew import Breast

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")



#!/usr/bin/env python
import subprocess
import os
 
def run(probability:float):
    """
    Run the Breast MRI Crew with model probability
    """
    # Get the absolute path to the streamlit_app.py file
    script_path = os.path.join(os.path.dirname(__file__), 'streamlit_app.py')

    # Command to run the Streamlit app
    command = ["streamlit", "run", script_path]

    inputs = {
        "probability":f"{probability:.4f}"
    }

    try:
        result = Breast().crew().kickoff(inputs=inputs)
        return result
    
    except Exception as e:
        raise Exception(f"Crew execution failed: {e}")


