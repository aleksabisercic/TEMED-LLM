import pandas as pd

from temed_llm.nlp.chat_model import create_llm
from temed_llm.data.data_schema import load_schema
from temed_llm.data.prompts import load_prompt
from temed_llm.core.rextract import RExtract
from temed_llm.utils.utils import setup_logger
from temed_llm.core.vorc import VORC
# from temed_llm.utils.config import dataset_name, openai_api_key

logger = setup_logger("INFO")


def main(dataset_names):
    for dataset_name in dataset_names:
        llm = create_llm("meta-llama/Llama-2-70b-chat-hf")
        temp = pd.read_csv(dataset_name)
        sample_size = min(150, len(temp))
        df = temp.sample(sample_size)

        # Load the appropriate prompt & schema for the given dataset_name
        pydantic_schema = load_schema(dataset_name)
        vorc_object = VORC(pydantic_object=pydantic_schema)

        prompt = load_prompt(dataset_name)
        prompt = prompt.replace("{pydantic_output_parser}", vorc_object.get_format_instructions())

        processor = RExtract(llm, vorc_object, prompt)

        checkpoint_file = f"{dataset_name.split('.')[0]}_checkpoint.csv"  # you may also prepend a directory if needed, like ./checkpoints/

        processor.process_dataframe(df, checkpoint_file)


if __name__ == "__main__":
    # Call the function with the desired dataset names
    main(["treatment.csv", "heart.csv"])  # "stroke.csv",
