from pydantic import BaseModel, Field, validator
from typing import List


class ContractChangeSummary(BaseModel):
    added_sections: List[str] = Field(
        ..., min_items=0, description="Contract sections that were added"
    )
    removed_sections: List[str] = Field(
        ..., min_items=0, description="Contract sections that were removed"
    )
    modified_sections: List[str] = Field(
        ..., min_items=0, description="Contract sections that were modified"
    )
    summary_of_the_change: str = Field(
        ..., min_length=5, description="Summary of the change with format Section X: -change_1 \n change_2, ..."
    )

    @validator("added_sections", "removed_sections", "modified_sections")
    def no_empty_strings(cls, v):
        if any(len(item.strip()) == 0 for item in v):
            raise ValueError("List items must not be empty")
        return v
