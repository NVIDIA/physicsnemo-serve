# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public configuration for a whole ensemble on one GPU."""

from datetime import datetime, timezone
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    field_validator,
    model_validator,
)

MAX_SEED = 2**63 - 1
NUMERICAL_POLICY = "earth2studio-native-v1"
RNG_POLICY = "torch-seed-once-native-v1"
MemberId = Annotated[StrictInt, Field(ge=1, le=50)]
Variable = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$")]


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    start_time: str
    ifs_member_id: MemberId | None = None
    nsteps: Annotated[StrictInt, Field(ge=1, le=60)]
    nensemble: Annotated[StrictInt, Field(ge=1, le=16)]
    batch_size: Annotated[StrictInt, Field(ge=1, le=16)] | None = None
    seed_base: Annotated[StrictInt, Field(ge=0, le=MAX_SEED)] = 1000
    output_variables: list[Variable] = Field(
        default_factory=lambda: ["t2m", "msl", "u10m", "v10m"], min_length=1
    )

    @field_validator("start_time")
    @classmethod
    def utc_cycle(cls, value: str) -> str:
        date = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if date.tzinfo is None:
            raise ValueError("start_time must include a UTC offset")
        date = date.astimezone(timezone.utc)
        if date.hour % 6 or date.minute or date.second or date.microsecond:
            raise ValueError("start_time must be a six-hour UTC initialization cycle")
        return date.isoformat().replace("+00:00", "Z")

    @model_validator(mode="after")
    def ensemble_contract(self):
        self.batch_size = min(self.nensemble, self.batch_size or self.nensemble)
        if len(set(self.output_variables)) != len(self.output_variables):
            raise ValueError("output_variables must be unique")
        return self
