from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from scalar_fastapi.scalar_fastapi import Theme, get_scalar_api_reference


class DocsRouter:
    def setup_routes(self, app: "FastAPI"):
        async def scalar_html():
            return get_scalar_api_reference(
                openapi_url=app.openapi_url,  # type: ignore
                title=app.title or "DocuMindSS API",
                theme=Theme.DEEP_SPACE,
                servers=[
                    {
                        "url": "http://localhost:8000",
                        "description": "Development server",
                    },
                ],
            )

        async def root_redirect():
            return RedirectResponse(url="/docs", status_code=302)

        app.add_api_route(
            "/docs", scalar_html, methods=["GET"], include_in_schema=False
        )
        app.add_api_route("/", root_redirect, methods=["GET"], include_in_schema=False)
