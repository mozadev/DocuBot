"""
DocuBot AI — Interfaz Streamlit.
Solo UI y presentación. Toda la lógica vive en domain/services/.
"""

import os
import tempfile
from typing import Dict, Any

import streamlit as st

from config.settings import settings
from api.factory import create_services

# --------- Config ----------
st.set_page_config(page_title="DocuBot AI", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main-header { font-size: 3rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 2rem; }
    .chat-message { padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; }
    .user-message { background-color: #e3f2fd; border-left: 4px solid #2196f3; }
    .bot-message { background-color: #f3e5f5; border-left: 4px solid #9c27b0; }
    .source-info { background-color: #fff3e0; padding: 0.5rem; border-radius: 0.25rem; font-size: 0.8rem; margin-top: 0.5rem; }
    .confidence-bar { background-color: #e0e0e0; border-radius: 0.25rem; height: 0.5rem; margin: 0.5rem 0; }
    .confidence-fill { background-color: #4caf50; height: 100%; border-radius: 0.25rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def init():
    try:
        return create_services()
    except Exception as e:
        st.error(f"Error inicializando: {e}")
        return None, None


def display_message(message: Dict[str, Any], is_user: bool = False):
    if is_user:
        st.markdown(f'<div class="chat-message user-message"><strong>👤 Tú:</strong><br>{message.get("question","")}</div>', unsafe_allow_html=True)
        return

    st.markdown(f'<div class="chat-message bot-message"><strong>🤖 DocuBot:</strong><br>{message.get("answer","")}</div>', unsafe_allow_html=True)

    if message.get("confidence", 0) > 0:
        pct = min(float(message["confidence"]) * 100, 100)
        st.markdown(f'<div class="confidence-bar"><div class="confidence-fill" style="width:{pct:.1f}%"></div></div><small>Confianza: {pct:.1f}%</small>', unsafe_allow_html=True)

    if message.get("sources"):
        with st.expander("📚 Ver fuentes"):
            for i, src in enumerate(message["sources"]):
                ct = src.get("content_type", "text")
                fname = src.get("filename", "Desconocido")
                content = src.get("content", "")
                score = src.get("score", 0.0)
                img_path = src.get("image_path", "")

                if ct == "image" and img_path and os.path.exists(img_path):
                    st.markdown(f'<div class="source-info"><strong>🖼️ Fuente {i+1} (imagen):</strong> {fname} | Score: {score:.3f}</div>', unsafe_allow_html=True)
                    st.image(img_path, caption=f"Imagen de {fname}", use_container_width=True)
                    with st.expander("📝 Descripción"):
                        st.markdown(content)
                elif ct == "mcp":
                    st.markdown(f'<div class="source-info" style="border-left:3px solid #2196f3;"><strong>🔌 Fuente {i+1} (MCP):</strong> {fname}<br>{content}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="source-info"><strong>📄 Fuente {i+1}:</strong> {fname}<br><strong>Contenido:</strong> {content}<br><strong>Score:</strong> {score:.3f}</div>', unsafe_allow_html=True)


def main():
    st.markdown('<h1 class="main-header">🤖 DocuBot AI</h1>', unsafe_allow_html=True)
    badge = "🖼️ Multimodal" if settings.enable_multimodal else "📄 Solo Texto"
    st.markdown(f"### Agente LangGraph + RAG Multimodal + MCP | {badge}")

    doc_svc, chat_svc = init()
    if not doc_svc or not chat_svc:
        st.error("Error al inicializar. Verifica tu configuración.")
        return

    # ---- Sidebar ----
    with st.sidebar:
        st.header("⚙️ Configuración")
        st.metric("Documentos en BD", doc_svc.get_document_count())

        if st.button("🗑️ Limpiar Base de Datos"):
            doc_svc.clear_database()
            chat_svc.clear_memory()
            st.success("Limpiado")
            st.rerun()

        if st.button("🧠 Limpiar Memoria"):
            chat_svc.clear_memory()
            st.success("Memoria limpiada")
            st.rerun()

        st.header("ℹ️ Sistema")
        st.info(f"Modelo: {settings.openai_model}")
        st.info(f"Embeddings: {settings.embedding_model}")
        st.success("🧠 Agente: LangGraph")
        if settings.enable_multimodal:
            st.success(f"🖼️ Vision: {settings.vision_model}")

        st.header("🔌 MCP")
        mcp_status = chat_svc.get_mcp_status()
        if mcp_status.connected_servers > 0:
            st.success(f"Conectados: {mcp_status.connected_servers}")
            for name in mcp_status.tool_names:
                st.text(f"  🔧 {name}")
        else:
            st.info("Sin servidores MCP")
        for srv in mcp_status.servers:
            st.caption(f"{'🟢' if srv.enabled else '⚪'} {srv.name}: {srv.description}")

    # ---- Tabs ----
    tab1, tab2, tab3 = st.tabs(["📄 Subir Documentos", "💬 Chat", "📊 Análisis"])

    with tab1:
        st.header("📄 Procesamiento de Documentos")
        exts = doc_svc.supported_extensions
        uploaded = st.file_uploader(f"Archivos soportados: {', '.join(exts)}", type=[e.lstrip('.') for e in exts], accept_multiple_files=True)

        if uploaded and st.button("🚀 Procesar"):
            with st.spinner("Procesando..."):
                total = 0
                for f in uploaded:
                    try:
                        suffix = f".{f.name.split('.')[-1].lower()}"
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(f.getvalue())
                            tmp_path = tmp.name

                        stats = doc_svc.process_and_index(tmp_path)
                        os.unlink(tmp_path)
                        total += stats["total"]

                        detail = f"{stats['text_chunks']} texto"
                        if stats["image_chunks"] > 0:
                            detail += f" + {stats['image_chunks']} imágenes"
                        st.success(f"✅ {f.name} ({detail})")
                    except Exception as e:
                        st.error(f"❌ {f.name}: {e}")
                st.success(f"🎉 Total: {total} chunks indexados")
                st.rerun()

    with tab2:
        st.header("💬 Chat Inteligente")
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for msg in st.session_state.chat_history:
            display_message(msg, is_user=msg.get("is_user", False))

        question = st.text_input("Pregunta:", placeholder="¿Qué dice el documento sobre...?", key="q")
        c1, c2 = st.columns([1, 4])
        with c1:
            if st.button("🚀 Enviar", type="primary") and question.strip():
                st.session_state.chat_history.append({"question": question, "is_user": True})
                with st.spinner("🤔 Pensando..."):
                    response = chat_svc.ask_question(question)
                st.session_state.chat_history.append(response.to_dict())
                st.rerun()
        with c2:
            if st.button("📝 Resumen"):
                with st.spinner("Generando..."):
                    st.info(f"**Resumen:**\n\n{chat_svc.get_conversation_summary()}")

    with tab3:
        st.header("📊 Análisis")
        c1, c2 = st.columns(2)
        with c1:
            count = doc_svc.get_document_count()
            st.metric("Total Chunks", count)
            st.success("✅ BD activa" if count > 0 else "⚠️ Sin documentos")
        with c2:
            mcp = chat_svc.get_mcp_status()
            for k, v in {
                "Motor": "LangGraph + MCP",
                "LLM": settings.openai_model,
                "Embeddings": settings.embedding_model,
                "Vision": settings.vision_model if settings.enable_multimodal else "Off",
                "MCP Servers": mcp.connected_servers,
                "MCP Tools": ", ".join(mcp.tool_names) or "Ninguna",
                "Chunk Size": settings.chunk_size,
            }.items():
                st.text(f"{k}: {v}")

        st.subheader("💭 Historial")
        if st.session_state.get("chat_history"):
            data = [
                {"Pregunta": m.get("question", ""), "Respuesta": (m.get("answer", "")[:100] + "..."), "Confianza": f"{m.get('confidence', 0):.2f}"}
                for m in st.session_state.chat_history if not m.get("is_user")
            ]
            if data:
                st.dataframe(data, use_container_width=True)
        else:
            st.info("Sin historial.")


if __name__ == "__main__":
    main()
