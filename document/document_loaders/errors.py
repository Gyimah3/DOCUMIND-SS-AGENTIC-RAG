class UnsupportedFileError(Exception):
    def __init__(self, file):
        self.message = f"Unsupported file type: {file}"
        super().__init__(self.message)
