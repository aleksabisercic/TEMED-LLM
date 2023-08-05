import json
import re
from pydantic import ValidationError
from typing import Any
import logging

logger = logging.getLogger(__name__)

class VORC(object):
    """
    VORC (Validation and Correction) module in the context of the TEMED-LLM system.

    This module is responsible for ensuring the accuracy and reliability of the information 
    extracted from medical texts. It detects and corrects errors that may occur during the 
    extraction process, such as JSON validation errors or typing validation errors. 

    If error correction is unsuccessful, it informs the LLM (large language model) about the mistake 
    and instructs it to correct the error, triggering the next loop iteration.

    Attributes:
        pydantic_object (Any): Pydantic model to validate the data against.
        feedback (str): Feedback message for error correction.

    Methods:
        fix_quotes(json_str: str) -> str: 
            Replace single quotes with double quotes and 'null' with 'None' in a JSON string.
        parse(text: str) -> dict: 
            Parse and validate a JSON string based on the pydantic_object schema.
        validation_feedback_loop(llm, message, max_retries=2) -> dict:
            Iteratively attempt to parse the result from a language model, providing feedback for error correction.
        get_format_instructions() -> str: 
            Generate instructions for formatting based on the pydantic_object schema.
    """
    pydantic_object: Any
    def __init__(self, pydantic_object):
        self.pydantic_object = pydantic_object
        self.feedback = """Reflect on provided error and correct the JSON object output"""
    
    def fix_quotes(self, json_str: str):
        """Fix JSON string by replacing single quotes with double quotes and 'null' with 'None'."""
        return json_str.replace("'", '"').replace("r'\bnull'", "None").replace("Null", "None")
        
    def parse(self, text: str):
        """
        Parse and validate a JSON string based on the pydantic_object schema.

        Extracts the first JSON object from the text and validates it against the pydantic schema. 
        Raises a json.JSONDecodeError or pydantic.ValidationError if the JSON object is not well formatted 
        or does not conform to the schema.
        """
        # Greedy search for 1st json candidate.
        match = re.search(
            "\{.*\}", text.strip(), re.MULTILINE | re.IGNORECASE | re.DOTALL
        )
        json_str = ""
        if match:
            json_str = self.fix_quotes(match.group())
        json_object = json.loads(json_str)
        self.pydantic_object.parse_obj(json_object)
        return json_object
    
    def validation_feedback_loop(self, llm, message, max_retries=2):
        """
        Iteratively attempt to parse the result from a language model, providing feedback for error correction.

        For max_retries times, it prompts the language model, attempts to parse the result, and 
        gives feedback in case of a json.JSONDecodeError or pydantic.ValidationError. If the error 
        persists after max_retries, it returns an empty dict.
        """
        logger.debug("Starting validation feedback loop...")
        for i in range(max_retries):
            try:
                answer = llm.forward(message)
                logger.debug(f"Answer from LLM: {answer}")
                answer_json = self.parse(answer)
                break  # if parsing was successful, break the retry loop
            except (json.JSONDecodeError, ValidationError) as e:
                logger.error(f"Error occurred when JSON was parsed: {str(e)}")
                if i < max_retries - 1:  # if not the last iteration
                    error_message = f"Error Occered when JSON was parsed: {str(e)}"
                    message = message + '\n' + answer + '\n' + error_message + '\n' + self.feedback
                    logger.debug(f"Prompting LLM with VORC message: {message}")
                else:  # if it's the last iteration
                    answer_json = {}  # return an empty dict if error persists
        logger.debug("Validation feedback loop completed.")
        return answer_json
            
    def get_format_instructions(self) -> str:
        schema = self.pydantic_object.schema()
        # Remove extraneous fields.
        reduced_schema = schema
        if "title" in reduced_schema:
            del reduced_schema["title"]
        if "type" in reduced_schema:
            del reduced_schema["type"]
        # Ensure json in context is well-formed with double quotes.
        schema_str = json.dumps(reduced_schema)
        return PYDANTIC_FORMAT_INSTRUCTIONS.format(schema=schema_str)

PYDANTIC_FORMAT_INSTRUCTIONS = """The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}}}
the object {{"foo": ["bar", "baz"]}} is a well-formatted instance of the schema. The object {{"properties": {{"foo": ["bar", "baz"]}}}} is not well-formatted.

Here is the output schema:
```
{schema}
```"""