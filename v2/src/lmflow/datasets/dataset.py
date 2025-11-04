#!/usr/bin/env python
# coding=utf-8
"""This Python code defines a class Dataset with methods for initializing, loading,
and manipulating datasets from different backends such as Hugging Face and JSON.
 
The `Dataset` class includes methods for loading datasets from a dictionary and a Hugging
Face dataset, mapping datasets, and retrieving the backend dataset and arguments.
"""



# Importing necessary libraries and modules
import copy
import json
import logging
from pathlib import Path

from cmath import e
from pathlib import Path
from typing import Optional, List

from datasets import load_dataset
from datasets import Dataset as HFDataset
from tqdm import tqdm

from lmflow.args import DatasetArguments
from lmflow.utils.constants import (
    DATASET_DESCRIPTION_MAP,
    TEXT_ONLY_DATASET_DESCRIPTION,
    TEXT2TEXT_DATASET_DESCRIPTION,
    FLOAT_ONLY_DATASET_DESCRIPTION,
    INSTANCE_FIELDS_MAP,
)
from lmflow.utils.versioning import is_multimodal_available
from lmflow.utils.data_utils import get_dataset_type_fast, check_dataset_instances_key_fast

if is_multimodal_available():
    from .multi_modal_dataset import CustomMultiModalDataset


logger = logging.getLogger(__name__)


DATASET_TYPES = [
    "text_only",
    "text2text",
    "float_only",
    "image_text",
    "conversation",
    "conversation_input_output",
    "paired_conversation",
    "paired_text_to_text",
    "text_to_textlist",
    "text_to_scored_textlist"
]

KEY_TYPE = "type"
KEY_INSTANCES = "instances"
KEY_SCORE = "score"

class Dataset:
    r"""
    Initializes the Dataset object with the given parameters.

    Parameters
    ------------
    data_args : DatasetArguments object.
        Contains the arguments required to load the dataset.

    backend : str,  default="huggingface"
        A string representing the dataset backend. Defaults to "huggingface".
    
    args : Optional.
        Positional arguments.
    
    kwargs : Optional.
        Keyword arguments.
    """
    def __init__(self, data_args: DatasetArguments=None, backend: str="huggingface", *args, **kwargs):
        self.data_args = data_args
        self.backend = backend
        self.backend_dataset = None
        self.type = None        # Original type of the dataset
        self.dataset_path = data_args.dataset_path

        if data_args.dataset_path is None:
            return

        if backend == "huggingface":
            # First, try Parquet support (commonly used for large-scale pretraining corpora like FineWeb)
            parquet_files = []
            dataset_path_obj = Path(self.dataset_path)
            if dataset_path_obj.is_file() and self.dataset_path.endswith('.parquet'):
                parquet_files = [dataset_path_obj.absolute().as_posix()]
            else:
                parquet_files = [
                    x.absolute().as_posix()
                    for x in dataset_path_obj.rglob("*.parquet")
                ]

            # Optionally limit the number of parquet files (e.g., for quick tests)
            if parquet_files and getattr(self.data_args, 'parquet_max_files', None):
                try:
                    max_n = int(self.data_args.parquet_max_files)
                    if max_n > 0:
                        parquet_files = sorted(parquet_files)[:max_n]
                except Exception:
                    pass

            if parquet_files:
                logger.info(f"Detected Parquet files (count={len(parquet_files)}). Loading via datasets parquet builder...")
                raw_dataset = load_dataset(
                    "parquet",
                    data_files=parquet_files,
                    split="train",
                    cache_dir=data_args.dataset_cache_dir,
                )
                # Infer a minimal dataset type; parquet pretraining shards are typically text-only
                if self.type is None:
                    # Prefer 'text_only' when a 'text' column exists; otherwise default to 'text_only' and rely on downstream mapping
                    self.type = 'text_only' if 'text' in raw_dataset.column_names else 'text_only'
                # Drop empty texts which can cause zero-length sequences downstream
                if 'text' in raw_dataset.column_names:
                    try:
                        raw_dataset = raw_dataset.filter(lambda x: isinstance(x['text'], str) and len(x['text']) > 0)
                    except Exception:
                        pass
                self.backend_dataset = raw_dataset
                self._check_instance_format()
                return

            # Fallback to JSON/JSONL support
            json_files = [
                x.absolute().as_posix()
                 for x in Path(self.dataset_path).glob("*.json")
            ]
            jsonl_files = [
                x.absolute().as_posix()
                 for x in Path(self.dataset_path).glob("*.jsonl")
            ]
            data_files = json_files + jsonl_files
            logger.info(f"Data files: \n{data_files}")
            
            if not data_files:
                raise ValueError(f"No .parquet, .json or .jsonl files found in {self.dataset_path}")
            
            # check if the dataset is in the correct format and get the dataset type (text_only, text2text, etc.)
            self._check_hf_json_format(data_files)
            # Load the dataset using the HuggingFace dataset library
            logger.info('Loading datasets')
            
            # Determine the file extension based on the files found
            if jsonl_files and not json_files:
                extensions = "json"
                # For jsonl files, we need to load them differently
                raw_dataset = load_dataset(
                    extensions,
                    data_files=data_files,
                    split="train",
                    cache_dir=data_args.dataset_cache_dir,
                )
            else:
                extensions = "json"
                raw_dataset = load_dataset(
                    extensions,
                    data_files=data_files,
                    field=KEY_INSTANCES,
                    split="train",
                    cache_dir=data_args.dataset_cache_dir,
                )
            self.backend_dataset = raw_dataset
            self._check_instance_format()
        elif backend == "json":
            # TODO (@Jiachun)
            pass
        elif backend == "custom_multi_modal":
            # FIXME refactor the backend name
            if not is_multimodal_available():
                raise ValueError(
                    'Multimodal not available. Please install via `pip install -e ".[multimodal]"`'
                )
            raw_dataset = CustomMultiModalDataset(self.dataset_path, data_args)
            self.backend_dataset = raw_dataset
        else:
            raise NotImplementedError(f'Unsupported dataset backend "{backend}"')

    
    def __len__(self):
        return len(self.backend_dataset)
    

    def _check_instance_format(self):
        """
        Checks if data (instances) have required fields. 
        Raises messages with hints if not matched.
        """
        fields = self.backend_dataset.features
        correct_fields = INSTANCE_FIELDS_MAP[self.type]
        if not set(correct_fields).issubset(set(fields)):
            raise ValueError(
                f'data instance fields incorrect'
                f' {list(correct_fields)} are required.'
            )
            
    
    def _check_hf_json_format(self, data_files: List[str]):
        for single_file in tqdm(data_files, desc='Checking dataset keys'):
            # Check if this is a jsonl file
            is_jsonl = single_file.endswith('.jsonl')
            
            if is_jsonl:
                # For jsonl files, infer the type from the data structure
                jsonl_data_type = self._infer_jsonl_type(single_file)
                if not jsonl_data_type:
                    raise ValueError(
                        f'Could not infer dataset type from jsonl file {single_file}. '
                        f'Please ensure the file contains valid data with recognizable fields.'
                    )
                if self.type is None:
                    self.type = jsonl_data_type
                elif self.type != jsonl_data_type:
                    raise ValueError(
                        'All task files must have same data types. Previous'
                        f' files have type "{self.type}", but in file'
                        f' {single_file}, it has type "{jsonl_data_type}".'
                    )
            else:
                # For regular json files, use the existing logic
                json_data_type = get_dataset_type_fast(single_file)
                if not json_data_type:
                    raise ValueError(
                        f'"{KEY_TYPE}" must be provided to initialize a dataset,'
                        f' e.g.\n'
                        f'    {TEXT_ONLY_DATASET_DESCRIPTION}'
                    )
                if self.type is None:
                    self.type = json_data_type
                elif self.type != json_data_type:
                    raise ValueError(
                        'All task files must have same data types. Previous'
                        f' files have type "{self.type}", but in file'
                        f' {single_file}, it has type "{self.type}".'
                    )
                # check if instances key is provided for json files
                key_instances_exists_flag = check_dataset_instances_key_fast(single_file, KEY_INSTANCES)
                if not key_instances_exists_flag:
                    raise ValueError(
                        f'"{KEY_INSTANCES}" must be provided to initialize a'
                        f' dataset, e.g.\n'
                        f'    {TEXT_ONLY_DATASET_DESCRIPTION}'
                    )

    def _infer_jsonl_type(self, file_path: str) -> str:
        """
        Infer the dataset type from a jsonl file by examining the structure of the first few lines.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Read first few lines to infer type
                for i, line in enumerate(f):
                    if i >= 5:  # Check first 5 lines
                        break
                    try:
                        data = json.loads(line.strip())
                        fields = set(data.keys())
                        
                        # Check for different dataset types based on field patterns
                        if 'input' in fields and 'output' in fields:
                            # Check if input is a list (conversation format) or string (text2text format)
                            if isinstance(data['input'], list) and len(data['input']) > 0:
                                if isinstance(data['input'][0], dict) and 'role' in data['input'][0]:
                                    # This is conversation format with role-based messages
                                    return 'conversation_input_output'
                            return 'text2text'
                        elif 'messages' in fields:
                            return 'conversation'
                        elif 'text' in fields and len(fields) == 1:
                            return 'text_only'
                        elif 'prompt' in fields and 'chosen' in fields and 'rejected' in fields:
                            return 'paired_text_to_text'
                        elif 'chosen' in fields and 'rejected' in fields:
                            return 'paired_conversation'
                        elif 'images' in fields and 'text' in fields:
                            return 'image_text'
                        elif 'value' in fields and len(fields) == 1:
                            return 'float_only'
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.warning(f"Error reading jsonl file {file_path}: {e}")
        
        return None


    def from_dict(self, dict_obj: dict, *args, **kwargs):
        r"""
        Create a Dataset object from a dictionary.

        Return a Dataset given a dict with format:
            {
                "type": TYPE,
                "instances": [
                    {
                        "key_1": VALUE_1.1,
                        "key_2": VALUE_1.2,
                        ...
                    },
                    {
                        "key_1": VALUE_2.1,
                        "key_2": VALUE_2.2,
                        ...
                    },
                    ...
                ]
            }

        Parameters
        -----------

        dict_obj : dict.
            A dictionary containing the dataset information.
        
        args : Optional.
            Positional arguments.
        
        kwargs : Optional.
            Keyword arguments.

        Returns
        ---------

        self : Dataset object.
        """
        if self.backend == "huggingface":
            if KEY_TYPE not in dict_obj:
                raise ValueError(
                    f'"{KEY_TYPE}" must be provided to initialize a dataset,'
                    f' e.g.\n'
                    f'    {TEXT_ONLY_DATASET_DESCRIPTION}'
                )
            if KEY_INSTANCES not in dict_obj:
                raise ValueError(
                    f'"{KEY_INSTANCES}" must be provided to initialize a'
                    f' dataset, e.g.\n'
                    f'    {TEXT_ONLY_DATASET_DESCRIPTION}'
                )

            self.type = dict_obj[KEY_TYPE]
            if not self.type in INSTANCE_FIELDS_MAP:
                raise ValueError(f'type "{self.type}" is not supported')

            correct_fields = INSTANCE_FIELDS_MAP[self.type]

            for i, instance in enumerate(dict_obj[KEY_INSTANCES]):
                fields = instance.keys()
                if not set(correct_fields).issubset(set(fields)):
                    raise ValueError(
                        f'data instance fields incorrect'
                        f' {list(correct_fields)} are required.'
                    )

            try:
                hf_dict = {}
                if len(dict_obj[KEY_INSTANCES]) > 0:
                    for key in dict_obj[KEY_INSTANCES][0].keys():
                        hf_dict[key] = [
                            instance[key] for instance in dict_obj[KEY_INSTANCES]
                        ]

                self.backend_dataset = HFDataset.from_dict(hf_dict, *args, **kwargs)
            except AttributeError as ex:
                raise ValueError(
                    f"Error occurs: {ex}. Failed to convert dict to"
                    f" \"{self.type}\" dataset," f" the standard format is as"
                    f" follows:\n"
                    f"    {DATASET_DESCRIPTION_MAP[self.type]}"
                )
            self._check_instance_format()

            return self
        elif self.backend == "dict":
            self.backend_dataset = dict_obj
            self.type = dict_obj[KEY_TYPE]
            return self
        else:
            raise NotImplementedError(
                f'Currently .from_dict is not supported for backend "{self.backend}"'
            )


    @classmethod
    def create_from_dict(cls, dict_obj, *args, **kwargs):
        r"""
        Returns
        --------

        Returns a Dataset object given a dict.
        """
        empty_data_args = DatasetArguments(dataset_path=None)
        dataset = Dataset(empty_data_args)
        return dataset.from_dict(dict_obj)


    def to_dict(self):
        r"""
        Returns
        ---------

        Return a dict represents the dataset:
            {
                "type": TYPE,
                "instances": [
                    {
                        "key_1": VALUE_1.1,
                        "key_2": VALUE_1.2,
                        ...
                    },
                    {
                        "key_1": VALUE_2.1,
                        "key_2": VALUE_2.2,
                        ...
                    },
                    ...
                ]
            }

        A python dict object represents the content of this dataset.
        """
        if self.backend == "huggingface":
            dict_obj = {}
            dict_obj[KEY_TYPE] = self.get_type()
            hf_dict = self.backend_dataset.to_dict()
            dict_obj[KEY_INSTANCES] = []

            first_key = None
            for key in hf_dict.keys():
                first_key = key
                break

            if first_key is not None:
                num_instances = len(hf_dict[first_key])
                dict_obj[KEY_INSTANCES] = [
                    {
                        key: hf_dict[key][i] for key in hf_dict.keys()
                    }
                    for i in range(num_instances)
                ]

            return dict_obj
        elif self.backend == "dict":
            dict_obj = self.backend_dataset
            return dict_obj
        else:
            raise NotImplementedError(
                f'Current .to_dict is not supported for backend "{self.backend}"'
            )


    def to_list(self):
        """Returns a list of instances."""
        if self.backend == "huggingface":
            instance_list = [self.backend_dataset.__getitem__(idx)
                             for idx in range(len(self.backend_dataset))]
            return instance_list
        elif self.backend == "dict":
            instance_list = copy.deepcopy(self.backend_dataset[KEY_INSTANCES])
            # TODO: should be a list of instances, instance should be huggingface datasets row format
            return instance_list
        else:
            raise NotImplementedError(
                f'Current .to_list is not supported for backend "{self.backend}"'
            )


    def map(self, *args, **kwargs):
        r"""
        Parameters
        ------------
        args : Optional.
            Positional arguments.
        
        kwargs : Optional.
            Keyword arguments.

        Returns
        ---------

        self : Dataset object.
        """
        # If the dataset uses Hugging Face as the backend, 
        # call the `map()` function of the Hugging Face backend dataset
        if self.backend == "huggingface":
            # Set the mapped dataset as the backend dataset of the current dataset
            mapped_backend_dataset = self.backend_dataset.map(*args, **kwargs)
            self.backend_dataset = mapped_backend_dataset
            return self
        else:
            # If the backend is not Hugging Face, raise a NotImplementedError
            raise NotImplementedError(
                f'Currently .map is not supported for backend "{self.backend}"'
            )


    def get_backend(self) -> Optional[str]:
        r"""
        Returns
        ---------

        self.backend
        """
        return self.backend


    def get_backend_dataset(self):
        r"""
        Returns
        ---------

        self.backend_dataset
        """
        return self.backend_dataset


    def get_fingerprint(self):
        r"""
        Returns
        ---------

        Fingerprint of the backend_dataset which controls the cache
        """
        return self.backend_dataset._fingerprint

    
    def get_data_args(self):
        r"""
        Returns
        ---------

        self.data_args
        """
        return self.data_args


    def get_type(self) -> str:
        r"""
        Returns
        ---------

        self.type
        """
        return self.type
    
    
    def save(
        self, 
        file_path: str, 
        format: str="json"
    ):
        r"""
        Save the dataset to a json file.

        Parameters
        ------------
        file_path : str.
            The path to the file where the dataset will be saved.
        """
        if format == "json":
            assert Path(file_path).suffix == ".json", "The file path must have a .json extension."
            with open(file_path, "w", encoding='utf-8') as fout:
                json.dump(self.to_dict(), fout, indent=4, ensure_ascii=False)
                
        else:
            logger.error(f"Unsupported format when saving the dataset: {format}.")
        
            
    def sample(self, n: int, seed: int=42):
        r"""
        Sample n instances from the dataset.

        Parameters
        ------------
        n : int.
            The number of instances to sample from the dataset.

        Returns
        ---------

        sample_dataset : Dataset object.
            A new dataset object containing the sampled instances.
        """
        if self.backend == "huggingface":
            sampled_dataset = self.backend_dataset.shuffle(seed=seed).select(range(n))
            output_dataset = self.create_from_dict(
                {
                    "type": self.get_type(),
                    "instances": [
                        {
                            col_name: sampled_dataset[col_name][i] for col_name in sampled_dataset.column_names
                        } for i in range(n)
                    ]
                }
            )
            return output_dataset
        else:
            raise NotImplementedError(
                f'Currently .sample is not supported for backend "{self.backend}"'
            )
            
            
    def train_test_split(self, test_size: float=0.2, shuffle: bool=True, seed: int=42):
        r"""
        Split the dataset into training and testing sets.

        Parameters
        ------------
        test_size : float, default=0.2.
            The proportion of the dataset that will be used for testing.

        Returns
        ---------

        train_dataset : Dataset object.
            A new dataset object containing the training instances.
        
        test_dataset : Dataset object.
            A new dataset object containing the testing instances.
        """
        if self.backend == "huggingface":
            splited = self.backend_dataset.train_test_split(
                test_size=test_size, shuffle=shuffle, seed=seed
            )
            train_dataset = self.create_from_dict(
                {
                    "type": self.get_type(),
                    "instances": [
                        {
                            col_name: splited["train"][col_name][i] for col_name in splited["train"].column_names
                        } for i in range(len(splited["train"]))
                    ]
                }
            )
            test_dataset = self.create_from_dict(
                {
                    "type": self.get_type(),
                    "instances": [
                        {
                            col_name: splited["test"][col_name][i] for col_name in splited["test"].column_names
                        } for i in range(len(splited["test"]))
                    ]
                }
            )
            return train_dataset, test_dataset
        else:
            raise NotImplementedError(
                f'Currently .train_test_split is not supported for backend "{self.backend}"'
            )
            
            
    def drop_instances(self, indices: list):
        r"""
        Drop instances from the dataset.

        Parameters
        ------------
        indices : list.
            A list of indices of the instances to drop from the dataset.
        """
        if self.backend == "huggingface":
            self.backend_dataset = self.backend_dataset.remove_indices(indices)
        else:
            raise NotImplementedError(
                f'Currently .drop_instances is not supported for backend "{self.backend}"'
            )
            
    
    def sanity_check(
        self, 
        drop_invalid: bool=True,
    ):
        r"""
        Perform a sanity check on the dataset.
        """
        if self.backend == "huggingface":
            self.hf_dataset_sanity_check(drop_invalid)
        else:
            raise NotImplementedError(
                f'Currently .sanity_check is not supported for backend "{self.backend}"'
            )
            
            
    def hf_dataset_sanity_check(
        self,
        drop_invalid: bool=True,
    ):
        r"""
        Perform a sanity check on the HuggingFace dataset.
        """
        if self.backend_dataset is None or len(self.backend_dataset) == 0:
            raise ValueError("Dataset is empty.")

        if self.type == 'text_to_textlist':
            num_output_per_instance = len(self.backend_dataset['output'][0])
            dataset_cache = self.backend_dataset.filter(lambda x: len(x['input'])!=0)
            dataset_cache = self.backend_dataset.filter(lambda x: len(x['output']) == num_output_per_instance)
            dataset_cache = self.backend_dataset.filter(lambda x: not all([len(output) == 0 for output in x['output']]))
            
            if len(dataset_cache) != len(self.backend_dataset):
                warning_info = (
                    f"Found {len(self.backend_dataset) - len(dataset_cache)} invalid instances "
                    "during hf_dataset_sanity_check, please check:\n"
                    "   1. length of input strings should not be empty\n"
                    "   2. length of output strings should not be all empty\n"
                    "   3. number of output strings should be consistent\n" # since we will use tensor reshape later
                )
                if drop_invalid:
                    self.backend_dataset = dataset_cache
                    logger.warning(warning_info+"Invalid instances are dropped.")
                else:
                    raise ValueError(warning_info)
        
        else:
            logger.warning(f"No sanity check for {self.type} dataset.")