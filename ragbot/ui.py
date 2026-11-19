"""Gradio-based chat interface for RAGBot.

Provides a professional, theme-aware UI with file upload, chat history,
example questions, and proper error handling.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import gradio as gr

from ragbot import __version__
from ragbot.config import get_settings
from ragbot.exceptions import RAGBotError
from ragbot.rag_chain import RAGChain

logger = logging.getLogger(__name__)

# Type alias for Gradio chat history (list of (user, bot) tuples).
ChatHistory = List[Tuple[str, str]]


class ChatUI:
    """Professional Gradio chat interface for RAGBot.

    Manages file uploads, chat state, and delegates question answering
    to :class:`RAGChain`.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._chain = RAGChain(self._settings)
        self._demo: Optional[gr.Blocks] = None
        self._uploaded_file: Optional[str] = None

    # ------------------------------------------------------------------
    # Gradio callbacks
    # ------------------------------------------------------------------

    def _on_file_upload(self, file) -> str:
        """Handle a new file upload.

        Args:
            file: Gradio file object (has a ``.name`` attribute).

        Returns:
            Status message shown beneath the upload widget.
        """
        if file is None:
            self._uploaded_file = None
            return "No file uploaded."
        try:
            file_path: str = file.name if hasattr(file, "name") else str(file)
            self._chain.ingest(file_path)
            self._uploaded_file = file_path
            return f"✅ **{file_path.split('/')[-1]}** ingested successfully."
        except RAGBotError as exc:
            logger.exception("Ingestion failed.")
            return f"❌ Ingestion error: {exc}"
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected ingestion error.")
            return f"❌ Unexpected error: {exc}"

    def _on_message(
        self,
        user_message: str,
        chat_history: ChatHistory,
    ) -> Tuple[ChatHistory, str]:
        """Process a user message and return the updated chat history.

        Args:
            user_message: Text typed by the user.
            chat_history: Current chat history.

        Returns:
            Tuple of (updated chat history, cleared textbox value).
        """
        chat_history = chat_history or []

        if not user_message.strip():
            return chat_history, ""

        if self._uploaded_file is None:
            bot_reply = "⚠️ Please upload a document before asking questions."
        else:
            try:
                bot_reply = self._chain.query(user_message)
            except RAGBotError as exc:
                logger.exception("Query failed.")
                bot_reply = f"❌ Error: {exc}"
            except Exception as exc:  # noqa: BLE001
                logger.exception("Unexpected query error.")
                bot_reply = f"❌ Unexpected error: {exc}"

        chat_history.append((user_message, bot_reply))
        return chat_history, ""

    def _on_clear(self) -> Tuple[ChatHistory, str]:
        """Clear chat history and the textbox.

        Returns:
            Tuple of (empty chat history, empty textbox value).
        """
        return [], ""

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def build(self) -> gr.Blocks:
        """Construct the Gradio Blocks UI.

        Returns:
            A fully-wired :class:`gr.Blocks` instance.
        """
        with gr.Blocks(theme=gr.themes.Soft(), title="RAGBot") as demo:
            # Header
            gr.Markdown(
                "# 🤖 RAGBot\n"
                "### Document-Grounded Conversational AI\n"
                "Upload a document and ask questions — answers are grounded "
                "in the uploaded content."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### 📄 Upload Document")
                    file_upload = gr.File(
                        label="Supported formats: PDF, TXT, CSV, DOCX",
                        file_types=[".pdf", ".txt", ".csv", ".docx"],
                    )
                    upload_status = gr.Markdown("No file uploaded.")

                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(label="Chat", height=450)
                    user_input = gr.Textbox(
                        placeholder="Ask a question about your document…",
                        show_label=False,
                        lines=1,
                    )
                    with gr.Row():
                        submit_btn = gr.Button("▶ Submit", variant="primary")
                        clear_btn = gr.Button("🗑 Clear")

            # Examples
            gr.Markdown("#### 💡 Example Questions")
            gr.Examples(
                examples=[
                    "Summarise the key points of this document.",
                    "What are the main conclusions?",
                    "List all the important dates mentioned.",
                ],
                inputs=user_input,
            )

            # Footer
            gr.Markdown(
                f"<center><sub>RAGBot v{__version__} · "
                f"Powered by LangChain & HuggingFace</sub></center>"
            )

            # --- Wiring ---
            file_upload.change(
                fn=self._on_file_upload,
                inputs=[file_upload],
                outputs=[upload_status],
            )

            submit_btn.click(
                fn=self._on_message,
                inputs=[user_input, chatbot],
                outputs=[chatbot, user_input],
            )

            user_input.submit(
                fn=self._on_message,
                inputs=[user_input, chatbot],
                outputs=[chatbot, user_input],
            )

            clear_btn.click(
                fn=self._on_clear,
                inputs=[],
                outputs=[chatbot, user_input],
            )

        self._demo = demo
        return demo

    # ------------------------------------------------------------------
    # Launch
    # ------------------------------------------------------------------

    def launch(self, **kwargs) -> None:
        """Build (if needed) and launch the Gradio application.

        Args:
            **kwargs: Extra keyword arguments forwarded to
                :meth:`gr.Blocks.launch`.
        """
        if self._demo is None:
            self.build()
        logger.info("Launching RAGBot UI…")
        self._demo.launch(**kwargs)  # type: ignore[union-attr]
