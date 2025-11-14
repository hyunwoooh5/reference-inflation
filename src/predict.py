from typing import Literal
from pydantic import BaseModel, Field

import pandas as pd


class Paper(BaseModel):
    number_of_pages: int = Field(..., ge=0)
    preprint_date: str
    author_count: int = Field(..., ge=0)
    document_type: Literal["article",
                           "conference paper", "book chapter", "thesis"]
    publication_type: Literal["research", "review", "lectures"]


def predict_single(model, paper: Paper):
    # Pydantic to dictionary
    paper_dict = paper.model_dump()

    # Dictionary to single-row dataframe
    X_df = pd.DataFrame([paper_dict])

    result = model.predict(X_df)

    return float(result[0])
