"""Entry point for running RAGBot as a module: python -m ragbot"""

from ragbot.ui import ChatUI


def main() -> None:
    """Launch the RAGBot application."""
    app = ChatUI()
    app.launch()


if __name__ == "__main__":
    main()
