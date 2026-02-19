"""
Factory: ensambla todas las capas (adapters -> services -> agents).
Punto unico de configuracion. Aqui se decide que implementaciones usar.
"""

from config.settings import settings
from adapters.vectordb.lancedb_adapter import LanceDBAdapter
from adapters.llm.openai_adapter import OpenAIAdapter
from adapters.loaders.pdf_loader import PDFLoader
from adapters.loaders.docx_loader import DOCXLoader
from adapters.mcp.client import MCPManager
from domain.services.document_service import DocumentService
from domain.services.chat_service import ChatService
from domain.services.marketing_service import MarketingService
from adapters.search.tavily_adapter import TavilyAdapter
from adapters.image.dalle_adapter import DalleAdapter
from agents.graph import build_agent_graph
from agents.marketing_graph import build_marketing_agent_graph
from core.logger import logger

_cache = {}


def _build_adapters():
    """Crea los adapters base (reutilizables por todos los servicios)."""
    if "adapters" in _cache:
        return _cache["adapters"]

    vector_store = LanceDBAdapter(
        db_path=settings.lancedb_path,
        table_name=settings.lancedb_table,
        embedding_model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )

    llm = OpenAIAdapter(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        vision_model=settings.vision_model,
        temperature=settings.openai_temperature,
    )

    pdf_loader = PDFLoader(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        enable_images=settings.enable_multimodal,
        images_dir=settings.images_path,
        min_image_size=settings.min_image_size,
        describe_image_fn=llm.describe_image if settings.enable_multimodal else None,
    )

    docx_loader = DOCXLoader(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    mcp_manager = MCPManager()

    tavily = TavilyAdapter(api_key=settings.tavily_api_key)
    dalle = DalleAdapter(api_key=settings.openai_api_key)

    adapters = {
        "vector_store": vector_store,
        "llm": llm,
        "pdf_loader": pdf_loader,
        "docx_loader": docx_loader,
        "mcp_manager": mcp_manager,
        "tavily": tavily,
        "dalle": dalle,
    }
    _cache["adapters"] = adapters
    return adapters


def create_services():
    """
    Crea e inyecta todas las dependencias.
    Retorna (document_service, chat_service) para Streamlit.
    """
    a = _build_adapters()

    document_service = DocumentService(
        loaders={".pdf": a["pdf_loader"], ".docx": a["docx_loader"]},
        vector_store=a["vector_store"],
    )

    chat_service = ChatService(
        vector_store=a["vector_store"],
        llm_adapter=a["llm"],
        mcp_manager=a["mcp_manager"],
        graph_builder=build_agent_graph,
        session_id="streamlit_user",
    )

    logger.info("Factory: servicios Streamlit creados")
    return document_service, chat_service


def create_all_services(session_id: str = "api"):
    """
    Crea todos los servicios incluyendo marketing.
    Para FastAPI / consumo externo (NestJS).
    """
    a = _build_adapters()

    document_service = DocumentService(
        loaders={".pdf": a["pdf_loader"], ".docx": a["docx_loader"]},
        vector_store=a["vector_store"],
    )

    chat_service = ChatService(
        vector_store=a["vector_store"],
        llm_adapter=a["llm"],
        mcp_manager=a["mcp_manager"],
        graph_builder=build_agent_graph,
        session_id=session_id,
    )

    marketing_graph = build_marketing_agent_graph(
        vector_store=a["vector_store"],
        llm_adapter=a["llm"],
        mcp_manager=a["mcp_manager"],
        tavily_adapter=a["tavily"],
        dalle_adapter=a["dalle"],
    )

    marketing_service = MarketingService(
        llm_adapter=a["llm"],
        vector_store=a["vector_store"],
        marketing_graph=marketing_graph,
        mcp_manager=a["mcp_manager"],
    )

    logger.info("Factory: todos los servicios creados (incluyendo marketing)")
    return document_service, chat_service, marketing_service, a["dalle"]
