# TEMED-LLM: Medical Text to Tabular Data Conversion System

This Python project is based on the TEMED-LLM methodology presented in a NIPS paper. The objective is to transform medical text into tabular data, enhancing interpretability and diagnostic performance.

The system comprises three main modules:

1. **RExtract**: A module that uses a Large Language Model (LLM) to extract tabular data from medical text.
2. **VORC**: A module that ensures the reliability and accuracy of the extracted data by identifying and correcting errors.
3. **ML Analysis**: A module responsible for training an interpretable machine learning model using the extracted tabular data.

## Code Structure

Here is the structure of the core modules:

- `config.py`: Contains configuration variables such as the OpenAI API key and dataset name.
- `data_schema.py`: Contains Pydantic schemas for data validation.
- `ml_modul.py`: Contains the SklearnClassifier class. This class serves as a wrapper for several Scikit-Learn classification models, facilitating preprocessing, training, evaluation, and prediction.
- `prompts.py`: Contains prompts to guide the Large Language Model in processing the medical texts.
- `plots.py`: Plotting funcions
- `rextract.py`: Contains the RExtract class, responsible for processing medical texts and extracting structured data.
- `tabular_evaluation.py`: Some of the evalution utility funcions used.
- `vorc.py`: Contains the VORC class, which is responsible for validating and correcting the extracted data.
- `utils.py`: Contains utility functions and classes for the system.
- `nlp_classifiers.py`: NLP classiferes used for benchamrking
- `chat_model.py`: LLM module
- `main.py`: The entry point of the application.

## Usage

The main entry point of the system is the `main.py` script. You can run this script to start the text-to-tabular data conversion process.

Here is a sample command to run the script:

```bash
python main.py
```

After tabular data was extracted we would train a ML modul `ml_modul.py`

## Configuration

The `config.py` file contains the following configurations:

```python
openai_api_key = "sk-XXXXX"
dataset_name = "heart.csv"
```

Replace `"sk-XXXXX"` with your actual OpenAI API key, and `"heart.csv"` with the name of the dataset you want to process.

## RExtract Module
The RExtract module is defined in the `rextract.py` file and contains the RExtract class. This class uses a Large Language Model (LLM) to process medical texts and a VORC object to validate and correct the LLM output.

## VORC Module
The VORC module is defined in the `vorc.py` file and contains the VORC class. This class uses a Pydantic schema to validate the data extracted by the RExtract module and provides feedback for error correction.

For more details about these modules, please refer to the source code files `rextract.py` and `vorc.py`.

## ML Module
The `ml_modul.py` contains the `SklearnClassifier` class. This class serves as a wrapper for several Scikit-Learn classification models, facilitating preprocessing, training, evaluation, and prediction. The classifier can be configured for both binary and multiclass classification tasks.

Here is the overview of the code:
```python
class SklearnClassifier(object):
    ...
    # Class implementation
    ...
```
See `ml_modul.py` for the detailed implementation.


## Prompts and Schemas

The system uses prompts and schemas to interact with the users and validate data. This section will guide you on how to define these in TEMED-LLM.

### Prompts

Prompts are defined in the `prompts.py` script. Here's an example of a prompt definition:

```python
import importlib

def load_prompt(dataset_name):
    """
    Loads the corresponding prompt for the given dataset.

    This function imports the prompts module dynamically and attempts to load the prompt with
    the name derived from the dataset_name. The prompt name is assumed to be the dataset name
    without the file extension, appended with '_prompt'. 

    Args:
        dataset_name (str): The name of the dataset whose prompt is to be loaded.

    Returns:
        The prompt for the given dataset.

    Raises:
        ValueError: If no prompt is found for the given dataset name.
    """
    ...
    return prompt

# Example prompt for 'heart.csv' dataset
heart_prompt = \
    """
    [Prompt details...]
    """
```

Each prompt is a Python string. You can customize these strings according to your needs.

## Schemas
Schemas are defined in the `data_schema.py` script. Here's an example of a schema definition:

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, Union
import importlib

def load_schema(dataset_name):    
    """
    Loads the corresponding Pydantic schema for the given dataset.

    This function imports the data_schema module dynamically and attempts to load the schema with
    the name derived from the dataset_name. The schema name is assumed to be the dataset name
    without the file extension, appended with '_schema'. 

    Args:
        dataset_name (str): The name of the dataset whose schema is to be loaded.

    Returns:
        The Pydantic schema for the given dataset.

    Raises:
        ValueError: If no schema is found for the given dataset name.
    """
    ...
    return pydantic_schema

class HeartData(BaseModel):
    """
    A Pydantic model for the 'heart.csv' dataset.
    """
    # Schema details...

heart_schema = HeartData
```

Each schema is a Pydantic model that describes the data structure expected from the user's input. You can customize these models according to the specific needs