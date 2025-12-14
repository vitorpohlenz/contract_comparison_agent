from pydantic import BaseModel, Field, validator
from typing import List


class ContractChangeSummary(BaseModel):
    topics_touched: List[str] = Field(
        ..., min_items=1, description="Legal or business topics affected"
    )
    sections_changed: List[str] = Field(
        ..., min_items=1, description="Contract sections that were changed"
    )
    summary_of_the_change: str = Field(
        ..., min_length=5, description="Summary of the change with format Section X: -change_1 \n change_2, ..."
    )

class ContextualizedContract(BaseModel):
    original_contract_text: str = Field(
        ..., min_length=5, description="Text of the original contract just the text impacted by the amendment"
    )
    amendment_text: str = Field(
        ..., min_length=5, description="Text of the amendment"
    )
