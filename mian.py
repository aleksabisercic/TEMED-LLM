import pandas as pd
from chat_model import create_llm
from data_schema import load_schema
from prompts import load_prompt
from vorc import VORC
from rextract import RExtract
from config import openai_api_key, dataset_name
from utils import setup_logger

logger = setup_logger("INFO")


def main():
    llm = create_llm(openai_api_key=openai_api_key)
    df = pd.read_csv(dataset_name)

    # Load the appropriate prompt & schema for the given dataset_name
    pydantic_schema = load_schema(dataset_name)
    vorc_object = VORC(pydantic_object=pydantic_schema)

    prompt = load_prompt(dataset_name)
    prompt = prompt.replace("{pydantic_output_parser}", vorc_object.get_format_instructions())

    processor = RExtract(llm, vorc_object, prompt)

    checkpoint_file = f"{dataset_name}_checkpoint.csv"  # ./checkpoints/

    processor.process_dataframe(df, checkpoint_file)


if __name__ == "__main__":
    main()
