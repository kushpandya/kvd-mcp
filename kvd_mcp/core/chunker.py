# src/kvd_mcp/core/chunker.py
import re
from typing import List
from dataclasses import dataclass

from .models import Note, Chunk

@dataclass
class ChunkerConfig:
    """Configuration for the chunking strategy."""
    max_chunk_tokens: int = 400      # ~300 words. Leaves room for context packing.
    min_chunk_tokens: int = 80       # Don't create tiny, useless chunks.
    overlap_tokens: int = 50         # Overlap between consecutive chunks.
    respect_headings: bool = True    # Split on Markdown headings (H1, H2, H3)

class Chunker:
    """
    Splits a Note into semantic Chunks.
    
    This is a pure transformation layer:
    - Input: Note object (from storage.py)
    - Output: List[Chunk] objects (for embeddings.py)
    
    No database access. No AI models. Easily testable.
    """

    def __init__(self, config: ChunkerConfig = ChunkerConfig()):
        self.config = config

    def chunk_note(self, note: Note) -> List[Chunk]:
        """
        Main entry point. Splits a note into chunks based on the configured strategy.
        """
        if not note.body.strip():
            return []

        if self.config.respect_headings:
            return self._heading_aware_chunk(note)
        else:
            return self._simple_chunk(note)

    # =========================================================================
    # HEADING-AWARE CHUNKING (For Markdown)
    # =========================================================================

    def _heading_aware_chunk(self, note: Note) -> List[Chunk]:
        """
        Splits the note on Markdown headings (H1, H2, H3).
        
        Strategy:
        1. Parse the document into sections (split by headings)
        2. Track the heading hierarchy (e.g., "# Docker > ## Networking")
        3. If a section is too large, sub-split by paragraphs
        4. If a section is too small, merge with the next section
        5. Prepend heading path to each chunk for context
        """
        sections = self._split_by_headings(note.body)
        chunks = []
        heading_stack = []  # Tracks: ["# Docker", "## Networking"]
        
        for section in sections:
            if section['type'] == 'heading':
                # Update heading stack based on level
                level = section['level']
                heading_stack = heading_stack[:level - 1]
                heading_stack.append(section['text'])
                continue
            
            # It's a content section
            heading_path = " > ".join(heading_stack) if heading_stack else note.metadata.title
            section_text = section['text'].strip()
            section_tokens = self._estimate_tokens(section_text)
            
            if section_tokens > self.config.max_chunk_tokens:
                # Sub-split by paragraphs
                sub_chunks = self._split_by_paragraphs(
                    section_text, 
                    self.config.max_chunk_tokens,
                    self.config.overlap_tokens
                )
                for i, sub_text in enumerate(sub_chunks):
                    chunks.append(self._create_chunk(
                        note, len(chunks), 
                        f"{heading_path} (part {i+1})", 
                        sub_text
                    ))
            elif section_tokens < self.config.min_chunk_tokens and chunks:
                # Merge with previous chunk
                chunks[-1].text += "\n\n" + section_text
                chunks[-1].token_count = self._estimate_tokens(chunks[-1].text)
            else:
                chunks.append(self._create_chunk(
                    note, len(chunks), heading_path, section_text
                ))
        
        return chunks

    def _split_by_headings(self, text: str) -> List[dict]:
        """
        Splits text into sections based on Markdown headings.
        Returns a list of dicts: {'type': 'heading'|'content', ...}
        """
        sections = []
        lines = text.split('\n')
        current_section = {'type': 'content', 'text': ''}
        
        for line in lines:
            # Check if line is a heading
            heading_match = re.match(r'^(#{1,3})\s+(.+)$', line)
            
            if heading_match:
                # Save current section if it has content
                if current_section['text'].strip():
                    sections.append(current_section)
                
                # Start new heading section
                level = len(heading_match.group(1))
                heading_text = heading_match.group(2)
                sections.append({
                    'type': 'heading',
                    'level': level,
                    'text': heading_text
                })
                current_section = {'type': 'content', 'text': ''}
            else:
                current_section['text'] += line + '\n'
        
        # Add final section
        if current_section['text'].strip():
            sections.append(current_section)
        
        return sections

    def _split_by_paragraphs(self, text: str, max_tokens: int, overlap_tokens: int) -> List[str]:
        """
        Splits a large section into smaller chunks by paragraphs.
        Applies overlap to maintain context between chunks.
        """
        paragraphs = text.split('\n\n')
        chunks = []
        current_text = ""
        
        for para in paragraphs:
            para_tokens = self._estimate_tokens(para)
            
            if self._estimate_tokens(current_text) + para_tokens > max_tokens:
                if current_text:
                    chunks.append(current_text.strip())
                    # Apply overlap: keep last N tokens from previous chunk
                    if overlap_tokens > 0:
                        overlap_text = self._get_last_n_tokens(current_text, overlap_tokens)
                        current_text = overlap_text + "\n\n" + para
                    else:
                        current_text = para
                else:
                    current_text = para
            else:
                current_text += para + "\n\n"
        
        if current_text.strip():
            chunks.append(current_text.strip())
        
        return chunks

    # =========================================================================
    # SIMPLE CHUNKING (Fallback)
    # =========================================================================

    def _simple_chunk(self, note: Note) -> List[Chunk]:
        """
        Simple paragraph-based chunking without heading awareness.
        Used as a fallback or for non-Markdown files.
        """
        paragraphs = note.body.split('\n\n')
        chunks = []
        current_text = ""
        
        for para in paragraphs:
            if self._estimate_tokens(current_text) + self._estimate_tokens(para) > self.config.max_chunk_tokens:
                if current_text:
                    chunks.append(self._create_chunk(
                        note, len(chunks), note.metadata.title, current_text.strip()
                    ))
                current_text = para + "\n\n"
            else:
                current_text += para + "\n\n"
        
        if current_text.strip():
            chunks.append(self._create_chunk(
                note, len(chunks), note.metadata.title, current_text.strip()
            ))
        
        return chunks

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _create_chunk(self, note: Note, index: int, heading_path: str, text: str) -> Chunk:
        """Creates a Chunk object with deterministic ID."""
        return Chunk(
            chunk_id=f"{note.note_id}::{index}",
            note_id=note.note_id,
            index=index,
            heading_path=heading_path,
            text=text,
            token_count=self._estimate_tokens(text)
        )

    def _estimate_tokens(self, text: str) -> int:
        """
        Rough token estimation. 
        For production, use tiktoken or the model's tokenizer.
        Rule of thumb: 1 token ≈ 4 characters for English text.
        """
        return len(text) // 4

    def _get_last_n_tokens(self, text: str, n_tokens: int) -> str:
        """Extracts the last N tokens from text (for overlap)."""
        # Rough estimate: n_tokens * 4 characters
        char_count = n_tokens * 4
        if len(text) <= char_count:
            return text
        return text[-char_count:]