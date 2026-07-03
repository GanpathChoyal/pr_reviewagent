"""
Diff Parser for parsing git diff format into structured change data
"""
import logging
from typing import List, Dict, Optional
from pathlib import Path

from unidiff import PatchSet, PatchedFile
from unidiff.errors import UnidiffParseError

from models import ParsedDiff, FileChange, LineChange, ChangeType

logger = logging.getLogger(__name__)


class DiffParseError(Exception):
    """Custom exception for diff parsing errors"""
    pass


class DiffParser:
    """Parser for git diff format content"""
    
    # Language detection mapping based on file extensions
    LANGUAGE_MAP = {
        # Python
        '.py': 'python',
        '.pyx': 'python',
        '.pyi': 'python',
        
        # JavaScript/TypeScript
        '.js': 'javascript',
        '.jsx': 'javascript',
        '.ts': 'typescript',
        '.tsx': 'typescript',
        '.mjs': 'javascript',
        '.cjs': 'javascript',
        
        # Java
        '.java': 'java',
        '.class': 'java',
        
        # C/C++
        '.c': 'c',
        '.h': 'c',
        '.cpp': 'cpp',
        '.cxx': 'cpp',
        '.cc': 'cpp',
        '.hpp': 'cpp',
        '.hxx': 'cpp',
        
        # C#
        '.cs': 'csharp',
        
        # Go
        '.go': 'go',
        
        # Rust
        '.rs': 'rust',
        
        # Ruby
        '.rb': 'ruby',
        '.rbw': 'ruby',
        
        # PHP
        '.php': 'php',
        '.php3': 'php',
        '.php4': 'php',
        '.php5': 'php',
        '.phtml': 'php',
        
        # Shell
        '.sh': 'shell',
        '.bash': 'shell',
        '.zsh': 'shell',
        '.fish': 'shell',
        
        # Web
        '.html': 'html',
        '.htm': 'html',
        '.css': 'css',
        '.scss': 'scss',
        '.sass': 'sass',
        '.less': 'less',
        
        # Configuration
        '.json': 'json',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.toml': 'toml',
        '.xml': 'xml',
        '.ini': 'ini',
        '.cfg': 'ini',
        '.conf': 'config',
        
        # Documentation
        '.md': 'markdown',
        '.markdown': 'markdown',
        '.rst': 'restructuredtext',
        '.txt': 'text',
        
        # SQL
        '.sql': 'sql',
        
        # Docker
        'dockerfile': 'dockerfile',
        '.dockerfile': 'dockerfile',
        
        # Other
        '.r': 'r',
        '.R': 'r',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.scala': 'scala',
        '.clj': 'clojure',
        '.hs': 'haskell',
        '.elm': 'elm',
        '.dart': 'dart',
    }
    
    # Binary file extensions that should be skipped
    BINARY_EXTENSIONS = {
        # Images
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.ico', '.webp',
        # Documents
        '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
        # Archives
        '.zip', '.tar', '.gz', '.bz2', '.7z', '.rar',
        # Executables
        '.exe', '.dll', '.so', '.dylib', '.bin',
        # Media
        '.mp3', '.mp4', '.avi', '.mov', '.wav', '.flac',
        # Fonts
        '.ttf', '.otf', '.woff', '.woff2',
        # Other
        '.pyc', '.pyo', '.class', '.jar', '.war'
    }
    
    def __init__(self):
        """Initialize the diff parser"""
        pass
    
    def parse(self, diff_content: str) -> ParsedDiff:
        """
        Parse git diff content into structured format
        
        Args:
            diff_content: Raw git diff content as string
            
        Returns:
            ParsedDiff object with structured change information
            
        Raises:
            DiffParseError: If diff content cannot be parsed
        """
        if not diff_content or not diff_content.strip():
            logger.warning("Empty diff content provided")
            return ParsedDiff(files=[])
        
        try:
            # Parse using unidiff library
            patch_set = PatchSet(diff_content)
            
            # Convert to our internal format
            files = []
            for patched_file in patch_set:
                file_change = self._process_patched_file(patched_file)
                if file_change:  # Skip None results (e.g., binary files)
                    files.append(file_change)
            
            return ParsedDiff(files=files)
            
        except UnidiffParseError as e:
            error_msg = f"Failed to parse diff content: {str(e)}"
            logger.error(error_msg)
            raise DiffParseError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error parsing diff: {str(e)}"
            logger.error(error_msg)
            raise DiffParseError(error_msg)
    
    def _process_patched_file(self, patched_file: PatchedFile) -> Optional[FileChange]:
        """
        Process a single patched file from unidiff
        
        Args:
            patched_file: PatchedFile object from unidiff
            
        Returns:
            FileChange object or None if file should be skipped
        """
        # Determine file path (prefer target path, fallback to source)
        file_path = patched_file.target_file
        if file_path.startswith('b/'):
            file_path = file_path[2:]  # Remove 'b/' prefix
        elif file_path == '/dev/null':
            # File deletion case, use source path
            file_path = patched_file.source_file
            if file_path.startswith('a/'):
                file_path = file_path[2:]  # Remove 'a/' prefix
        
        # Check if file is binary
        if patched_file.is_binary_file or self._is_binary_file(file_path):
            logger.debug(f"Skipping binary file: {file_path}")
            return FileChange(
                file_path=file_path,
                language=self.detect_language(file_path),
                is_binary=True,
                additions=[],
                deletions=[],
                modifications=[]
            )
        
        # Detect language
        language = self.detect_language(file_path)
        
        # Extract changes
        additions, deletions, modifications = self.extract_changes(patched_file)
        
        return FileChange(
            file_path=file_path,
            language=language,
            is_binary=False,
            additions=additions,
            deletions=deletions,
            modifications=modifications
        )
    
    def detect_language(self, file_path: str) -> str:
        """
        Detect programming language from file path
        
        Args:
            file_path: Path to the file
            
        Returns:
            Language name as string, 'unknown' if not recognized
        """
        if not file_path:
            return 'unknown'
        
        # Convert to Path object for easier handling
        path = Path(file_path.lower())
        
        # Check for special filenames first
        filename = path.name
        if filename in ['dockerfile', 'makefile', 'rakefile', 'gemfile']:
            special_map = {
                'dockerfile': 'dockerfile',
                'makefile': 'makefile',
                'rakefile': 'ruby',
                'gemfile': 'ruby'
            }
            return special_map.get(filename, 'unknown')
        
        # Check file extension
        extension = path.suffix
        if extension in self.LANGUAGE_MAP:
            return self.LANGUAGE_MAP[extension]
        
        # Check for compound extensions
        if len(path.suffixes) >= 2:
            compound_ext = ''.join(path.suffixes[-2:])
            if compound_ext in self.LANGUAGE_MAP:
                return self.LANGUAGE_MAP[compound_ext]
        
        # Default to unknown
        return 'unknown'
    
    def _is_binary_file(self, file_path: str) -> bool:
        """
        Check if file is likely binary based on extension
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file is likely binary
        """
        if not file_path:
            return False
        
        path = Path(file_path.lower())
        extension = path.suffix
        
        return extension in self.BINARY_EXTENSIONS
    
    def extract_changes(self, patched_file: PatchedFile) -> tuple[List[LineChange], List[LineChange], List[LineChange]]:
        """
        Extract line changes from a patched file
        
        Args:
            patched_file: PatchedFile object from unidiff
            
        Returns:
            Tuple of (additions, deletions, modifications)
        """
        additions = []
        deletions = []
        modifications = []
        
        # Process each hunk in the file
        for hunk in patched_file:
            # Track line numbers for context
            source_line_no = hunk.source_start
            target_line_no = hunk.target_start
            
            # Process each line in the hunk
            for line in hunk:
                if line.is_added:
                    # Line addition
                    additions.append(LineChange(
                        line_number=target_line_no,
                        content=line.value.rstrip('\n\r'),
                        change_type=ChangeType.ADD
                    ))
                    target_line_no += 1
                    
                elif line.is_removed:
                    # Line deletion
                    deletions.append(LineChange(
                        line_number=source_line_no,
                        content=line.value.rstrip('\n\r'),
                        change_type=ChangeType.DELETE
                    ))
                    source_line_no += 1
                    
                elif line.is_context:
                    # Context line (unchanged)
                    source_line_no += 1
                    target_line_no += 1
        
        # Detect modifications (deleted line followed by added line)
        modifications = self._detect_modifications(deletions, additions)
        
        return additions, deletions, modifications
    
    def _detect_modifications(self, deletions: List[LineChange], additions: List[LineChange]) -> List[LineChange]:
        """
        Detect line modifications by matching nearby deletions and additions
        
        This is a heuristic approach - we consider a deletion followed by an addition
        within a small range as a potential modification.
        
        Args:
            deletions: List of deleted lines
            additions: List of added lines
            
        Returns:
            List of modifications (using the new line content)
        """
        modifications = []
        
        # Simple heuristic: if we have equal numbers of deletions and additions
        # in close proximity, treat them as modifications
        if len(deletions) == len(additions) and len(deletions) <= 10:
            # For small changes, assume they're modifications
            for addition in additions:
                modifications.append(LineChange(
                    line_number=addition.line_number,
                    content=addition.content,
                    change_type=ChangeType.MODIFY
                ))
        
        return modifications
    
    def get_file_stats(self, parsed_diff: ParsedDiff) -> Dict[str, int]:
        """
        Get statistics about the parsed diff
        
        Args:
            parsed_diff: ParsedDiff object
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_files': len(parsed_diff.files),
            'binary_files': sum(1 for f in parsed_diff.files if f.is_binary),
            'text_files': sum(1 for f in parsed_diff.files if not f.is_binary),
            'total_additions': sum(len(f.additions) for f in parsed_diff.files),
            'total_deletions': sum(len(f.deletions) for f in parsed_diff.files),
            'total_modifications': sum(len(f.modifications) for f in parsed_diff.files),
            'languages': {}
        }
        
        # Count files by language
        for file_change in parsed_diff.files:
            lang = file_change.language
            stats['languages'][lang] = stats['languages'].get(lang, 0) + 1
        
        return stats
    
    def filter_by_language(self, parsed_diff: ParsedDiff, languages: List[str]) -> ParsedDiff:
        """
        Filter parsed diff to only include files of specified languages
        
        Args:
            parsed_diff: ParsedDiff object
            languages: List of language names to include
            
        Returns:
            New ParsedDiff with filtered files
        """
        filtered_files = [
            file_change for file_change in parsed_diff.files
            if file_change.language in languages
        ]
        
        return ParsedDiff(files=filtered_files)
    
    def get_changed_lines_content(self, file_change: FileChange) -> List[str]:
        """
        Get all changed line content from a file change
        
        Args:
            file_change: FileChange object
            
        Returns:
            List of all changed line contents
        """
        content = []
        
        # Add all additions
        content.extend([line.content for line in file_change.additions])
        
        # Add all modifications (new content)
        content.extend([line.content for line in file_change.modifications])
        
        return content