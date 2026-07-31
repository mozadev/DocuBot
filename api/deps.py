"""FastAPI dependencies."""

from __future__ import annotations

from api.factory import Container, build_container


def get_container() -> Container:
    """
    Inject the application container into a route.

    build_container() is memoised, so this is a cheap lookup after startup.
    Going through a dependency rather than a module-level global is what lets a
    test override it with fakes via app.dependency_overrides.
    """
    return build_container()
