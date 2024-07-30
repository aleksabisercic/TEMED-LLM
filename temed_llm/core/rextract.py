import logging
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


class RExtract:
    """
    RExtract (Reason and Extract) module in the context of the TEMED-LLM system.

    for reasoning and extracting data from unstructured (medical) texts.

    Attributes:
        llm: A Large Language Model used for processing text.
        vorc_object: VORC: A VORC object used for validation and correction of model output.
        prompt (str): A prompt used to guide the language model in processing medical texts.

    Methods:
        process_text(text: str) -> Dict:
            Extract structured data from an unstructured (medical) text.
        process_dataframe(df: pd.DataFrame, checkpoint_file: str, checkpoint_freq=50):
            Process a DataFrame of medical texts and save results at specified intervals.
        save_checkpoint(df: pd.DataFrame, answers: List[Dict], index: int, checkpoint_file: str):
            Save intermediate results into a checkpoint file.
    """

    def __init__(self, llm, vorc_object, prompt):
        self.llm = llm
        self.vorc_object = vorc_object
        self.prompt = prompt

    def process_text(self, text: str) -> Dict:
        """
        Use the language model and the VORC module to extract and validate structured data.

        from a provided unstructured medical text.
        """
        logger.debug("Processing text...")
        message = self.prompt.replace("{medical_report_to_replace}", text)
        return self.vorc_object.validation_feedback_loop(self.llm, message, 2)

    def process_dataframe(self, df: pd.DataFrame, checkpoint_file: str, checkpoint_freq=50):
        """
        Process a DataFrame of medical texts and save results at specified intervals.

        For each row in the DataFrame, extract structured data from the 'medical_report' column,
        append the results to a list, and periodically save these results to a CSV file.
        """
        logger.info("Beginning to process DataFrame...")
        answers = []

        df = df.reset_index(drop=True)
        for index, row in df.iterrows():
            logger.info(f"Processing row {index}...")
            report_to_extract = row["medical_report"]
            answer_json = self.process_text(report_to_extract)
            answers.append(answer_json)

            if index % checkpoint_freq == 0:
                # Check the length of answers list
                if len(answers) != index + 1:
                    logger.warning(
                        f"Length mismatch! Answers length: {len(answers)}, DataFrame index: {index}"
                    )

                logger.info(f"Checkpointing at index {index}")
                self.save_checkpoint(df, answers, index, checkpoint_file)

        # Save the final results
        self.save_checkpoint(df, answers, index, checkpoint_file)
        logger.info(f"DataFrame processing complete and can be found at {checkpoint_file}")

    @staticmethod
    def save_checkpoint(df: pd.DataFrame, answers: List[Dict], index: int, checkpoint_file: str):
        checkpoint_df = df.iloc[: len(answers)].copy()
        checkpoint_df["answers"] = answers
        checkpoint_df.to_csv(checkpoint_file)
        logger.info(f"Checkpointing successful at: {checkpoint_file}")
