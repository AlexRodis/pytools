# Database data loading utilities
from dataclasses import dataclass, field
import typing
import pandas as pd
from django.db import models

@dataclass(kw_only=True, slots=True)
class FromFile:
    r'''
        Create a database table from a pandas dataframe
        WIP
    '''
    dataframe:typing.Optional[pd.DataFrame] = field(default=None)
    filepath:str = field(default=None)
    filetype:typing.Optional[str] = field(default=None)
    schema:typing.Optional[dict] = field(default=None)
    
    
    def __post_init__(self)->None:
        if self.filepath is None:
            raise ValueError((
                "`filepath` must be a string. Received "
                f"{self.filepath} instead"
                ))
        self.filetype = self.filepath.split(".")[-1]
        if self.filetype != "csv":
            raise TypeError((
                "Unsupported file type. Expected csv but saw "
                f"{self.filetype} instead"
            ))
        if self.dataframe is None:
            self.dataframe = pd.read_csv(self.filepath)
        if self.schema == None:
            self.dataframe = pd.infer_objects(self.dataframe)
            coredf: pd.DataFrame = self.dataframe
            self.schema = {
                colname: coredf[colname].dtype for colname in \
                    coredf.columns.to_list()
            }