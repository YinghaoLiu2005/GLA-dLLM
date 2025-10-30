# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from Dream repos: https://github.com/HKUNLP/Dream

"""Tokenization classes for GLADream. Reuses DreamTokenizer."""

# For simplicity, GLADream reuses the same tokenizer as Dream
# This is a wrapper that aliases DreamTokenizer
from ..dream.model.tokenization_dream import DreamTokenizer


class GLADreamTokenizer(DreamTokenizer):
    """GLADream tokenizer - same as Dream"""
    pass

