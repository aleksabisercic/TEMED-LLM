import json
import logging
import logging.config
import os
import sys

from langchain.schema import AIMessage, BaseMessage, ChatMessage, HumanMessage, SystemMessage

# SET LOGGING
logger = logging.getLogger(__name__)


def _convert_message_to_dict(message: BaseMessage) -> dict:
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    else:
        raise ValueError(f"Got unknown type {message}")
    if "name" in message.additional_kwargs:
        message_dict["name"] = message.additional_kwargs["name"]
    return message_dict


def get_num_tokens_from_text(text: str, model="gpt-3.5-turbo") -> int:
    """Calculate num tokens for gpt-3.5-turbo and gpt-4 with tiktoken package."""
    try:
        import tiktoken
    except ImportError:
        raise ValueError(
            "Could not import tiktoken python package. "
            "This is needed in order to calculate get_num_tokens. "
            "Please it install it with `pip install tiktoken`."
        )

    if model == "gpt-3.5-turbo":
        # gpt-3.5-turbo may change over time.
        # Returning num tokens assuming gpt-3.5-turbo-0301.
        model = "gpt-3.5-turbo-0301"
    elif model == "gpt-4":
        # gpt-4 may change over time.
        # Returning num tokens assuming gpt-4-0314.
        model = "gpt-4-0314"

    # Returns the number of tokens used by a list of messages.
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")

    if model == "gpt-3.5-turbo-0301":
        # every message follows <im_start>{role/name}\n{content}<im_end>\n
        tokens_per_message = 4
        # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
    else:
        raise NotImplementedError(
            f"get_num_tokens_from_messages() is not presently implemented "
            f"for model {model}."
            "See https://github.com/openai/openai-python/blob/main/chatml.md for "
            "information on how messages are converted to tokens."
        )
    num_tokens = 0
    # calculate number of tokens for given text
    num_tokens += len(encoding.encode(text))
    num_tokens += tokens_per_message
    # every reply is primed with <im_start>assistant
    num_tokens += 3
    return num_tokens


def remove_key_from_json(json_str, key_to_remove):
    """
    Remove a specified key from a JSON string.

    Args:
        json_str (str): The JSON string to modify.
        key_to_remove (str): The key to remove from the JSON string.

    Returns:
        str: The modified JSON string without the specified key.
    """
    try:
        # Load the JSON string into a Python dictionary
        data = json.loads(json_str)

        # Remove the key if it exists
        if key_to_remove in data:
            del data[key_to_remove]

        # Dump the Python dictionary back into a JSON string
        json_str_modified = json.dumps(data)

        return json_str_modified
    except json.JSONDecodeError:
        print(f"Invalid JSON string: {json_str}")
        return json_str


def prepare_json_row(row):
    json_row = row.to_json()
    return remove_key_from_json(json_str=json_row, key_to_remove="HeartDisease")


def save_checkpoint(df, answers, index, dataset_name):
    for i in range(index + 1):
        df.at[i, "answers"] = answers[i]
    df.loc[:index].to_csv(f"{dataset_name}_processed_{index}.csv", index=False)


def save_checkpoint_with_stats(
    df,
    medical_reports,
    total_tokens,
    completion_tokens,
    total_cost,
    time_taken,
    index,
    dataset_name,
):
    print(f"Mean tokens: {sum(total_tokens)/len(total_tokens)} for {len(total_tokens)} samples")
    print("Total cost: ", sum(total_cost))
    print("Mean time taken: ", sum(time_taken) / len(time_taken))
    print("Total time taken: ", sum(time_taken))
    print("\n")

    for i in range(index + 1):
        df.at[i, "medical_report"] = medical_reports[i]
        df.at[i, "total_tokens"] = total_tokens[i]
        df.at[i, "completion_tokens"] = completion_tokens[i]
        df.at[i, "time_taken"] = time_taken[i]

    df.loc[:index].to_csv(f"{dataset_name}_medical_report_{index}.csv", index=False)


def get_log_level():
    return os.getenv("LOG_LEVEL", "INFO").upper()


def load_required_env_var(key):
    env_var = os.getenv(key)

    if env_var is None:
        logger.error("Required environment variable: " + key + " is not defined.")
        raise KeyError(f"Required environment variable: {key}  is not set.")
    else:
        logger.debug("Loaded environment variable: " + key + " = " + env_var)

    return env_var


def load_env_var(key, default=None):
    env_var = os.getenv(key, default)

    if env_var is None:
        logger.debug("Environment variable: " + key + " is not defined.")
    else:
        logger.debug("Loaded environment variable: " + key + " = " + str(env_var))

    return env_var


def get_debug_level(debug_level):
    """Transform debug level from string to logging flags.

    Args:
        debug_level (str): Debug level as string.

    Returns:
        int: Debug level as logging flag.
    """
    if debug_level == "INFO":
        return logging.INFO
    elif debug_level == "DEBUG":
        return logging.DEBUG
    elif debug_level == "WARNING":
        return logging.WARNING
    elif debug_level == "ERROR":
        return logging.ERROR
    elif debug_level == "CRITICAL":
        return logging.CRITICAL
    else:
        return logging.INFO  # defults to INFO


def disable_existing_loggers(logger_name):
    """Disable existing loggers given a name.

    Args:
        logger_name (str): Logger name
    """
    log_dict = logging.root.manager.loggerDict
    for enabled_logger in log_dict:
        if logger_name in enabled_logger:
            log_dict[enabled_logger].disabled = True


def setup_logger(debug_level="ERROR"):
    """Setup logging configuration.

    To set up the logger, call this function in your main script.
    To get the logger in other modules, call ``log = logging.getLogger(__name__)`` in each module,
    it will automatically get the setup configuration.

    Args:
        debug_level (str): Debug level as string.
        config_file (str|bool): Yaml configuration file.

    Returns:
        obj: Logging object.

    Examples:
        >>> log = setup_logger(debug_level='DEBUG')#It will show: 2018-03-10 09:05:14 DEBUG [test.py:6]: Debug log_base
        >>> log.debug("Debug log_base") #doctest: +ELLIPSIS
        20... DEBUG [<doctest ...log_base.logger.setup_logger[1]>:1]: Debug log_base
        >>> log.info("Debug log_base") #doctest: +ELLIPSIS
        20... INFO [<doctest ...log_base.logger.setup_logger[2]>:1]: Debug log_base
        >>> log = setup_logger(debug_level='INFO', config_file='log_base/logging.yaml')
        >>> log.error("Error log_base") #doctest: +ELLIPSIS
        20... ERROR [<doctest ...log_base.logger.setup_logger[4]>:1]: Error log_base
        >>> log.debug("Debug log_base") #should return nothing because log level is set to info
        >>> os.environ['DEBUG_LEVEL'] = "DEBUG"
        >>> log = setup_logger(debug_level='INFO')
        >>> log.debug("Debug log_base") #doctest: +ELLIPSIS
        20... DEBUG [<doctest ...log_base.logger.setup_logger[8]>:1]: Debug log_base
    """
    level = get_debug_level(debug_level)

    # Get logger
    log = logging.getLogger()
    log.setLevel(level)

    # Format logger
    console = logging.StreamHandler(stream=sys.stdout)
    format_str = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)s]: %(message)s"
    format_time = "%Y-%m-%d %H:%M:%S"
    console.setFormatter(logging.Formatter(format_str, format_time))
    log.addHandler(console)

    return log
