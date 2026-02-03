"""Application entry point. Run with: uvicorn main:app --reload"""

from app.config import settings
from app.container import ApplicationContainer

container = ApplicationContainer(settings)
app = container.instance
