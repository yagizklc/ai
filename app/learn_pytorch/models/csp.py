from pydantic import BaseModel


class Dependency(BaseModel):
    name: str
    version: str
    dependencies: list['Dependency']

    @property
    def version_int(self):
        return self.version


class CSP:
    def __init__(self):
        pass
