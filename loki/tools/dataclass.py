# SPDX-FileCopyrightText: 2018 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileComment: In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from functools import partial

from pydantic.dataclasses import dataclass as dataclass_validated


__all__ = ['dataclass_strict']


# Configuration for validation mechanism via pydantic
dataclass_validation_config  = {
    'arbitrary_types_allowed': True,
}

# Using this decorator, we can force strict validation
dataclass_strict = partial(dataclass_validated, config=dataclass_validation_config)
