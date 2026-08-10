from pathlib import Path
from app.knowledge.models import SourceDocument

SUPPORTED_EXTENSIONS =  {".txt", ".md"}

def load_documents(directory: str) -> list[SourceDocument]:
    document_directory = Path(directory)
    
    if not document_directory.exists():
        raise FileNotFoundError(f"document directory does not exist: {directory}")
    
    if not document_directory.is_dir():
        raise NotADirectoryError(f"document path is not a directory: {directory}")
    
    documents: list[SourceDocument] = []
    
    for file_path in sorted (document_directory.iterdir()):
        if not file_path.is_file():
            continue
        
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        
        content = file_path.read_text(encoding="utf-8").strip()
        
        if not content:
            continue
        
        documents.append(
            SourceDocument(
                source=file_path.name,
                content=content,
            )
        )
        
    return documents