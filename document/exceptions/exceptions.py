class FileUnsupportedError(Exception):
    def __init__(self, message="File Unsupported", *args: object) -> None:
        super().__init__(message, *args)
