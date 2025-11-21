#!/usr/bin/env python3
"""
Data Collection Script for Nima - Applications Dataset.

This script collects documentation and code context from all user applications:
- canopy (personal finance dashboard)
- swimTO (Toronto pool schedules)
- us-law-severity-map (US law visualization)

It formats the data as Q&A pairs for training nima to understand these applications.
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ApplicationDataCollector:
    """Collects documentation and code context from applications."""
    
    def __init__(self, workspace_root: str):
        """
        Initialize collector.
        
        Args:
            workspace_root: Root directory containing all applications
        """
        self.workspace_root = Path(workspace_root)
        self.applications = {
            'canopy': {
                'path': self.workspace_root / 'canopy',
                'name': 'Canopy',
                'description': 'Self-hosted personal finance, investment, and budgeting dashboard',
                'docs': ['README.md', 'ARCHITECTURE.md', 'CHANGELOG.md', 'MASTER_PROMPT.md', 'CSV_IMPORT_GUIDE.md']
            },
            'swimTO': {
                'path': self.workspace_root / 'swimTO',
                'name': 'SwimTO',
                'description': 'Aggregates and displays indoor community pool drop-in swim schedules for Toronto',
                'docs': ['README.md', 'MASTER_PROMPT.md', 'PROJECT_STRATEGY.md']
            },
            'us-law-severity-map': {
                'path': self.workspace_root / 'us-law-severity-map',
                'name': 'US Law Severity Map',
                'description': 'Interactive choropleth map showing law severity scores and crime statistics for US states',
                'docs': ['README.md', 'PROMPT.md', 'CHANGELOG.md']
            }
        }
        
        # Files to extract code context from
        self.code_files = {
            'canopy': ['backend/api/*.py', 'backend/models/*.py', 'backend/app/*.py'],
            'swimTO': ['apps/api/**/*.py', 'data-pipeline/**/*.py'],
            'us-law-severity-map': ['main.py']
        }
    
    def read_file_safe(self, filepath: Path) -> str:
        """Safely read a file, returning empty string if not found."""
        try:
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
        except Exception as e:
            logger.warning(f"Could not read {filepath}: {e}")
        return ""
    
    def extract_markdown_sections(self, content: str) -> List[Dict[str, str]]:
        """
        Extract sections from markdown content.
        
        Returns:
            List of {title, content} dictionaries
        """
        sections = []
        
        # Split by headers
        header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        parts = header_pattern.split(content)
        
        current_title = "Introduction"
        current_content = []
        
        for i, part in enumerate(parts):
            if part.startswith('#'):
                # Save previous section
                if current_content:
                    sections.append({
                        'title': current_title,
                        'content': '\n'.join(current_content).strip()
                    })
                # Start new section
                current_title = part.strip('#').strip()
                current_content = []
            else:
                current_content.append(part)
        
        # Add final section
        if current_content:
            sections.append({
                'title': current_title,
                'content': '\n'.join(current_content).strip()
            })
        
        return sections if sections else [{'title': 'Content', 'content': content}]
    
    def collect_application_docs(self, app_key: str) -> List[Dict[str, str]]:
        """
        Collect documentation for an application.
        
        Returns:
            List of Q&A pairs
        """
        app_info = self.applications[app_key]
        app_path = app_info['path']
        
        if not app_path.exists():
            logger.warning(f"Application path does not exist: {app_path}")
            return []
        
        qa_pairs = []
        
        # Basic Q&A about the application
        qa_pairs.append({
            'question': f'What is {app_info["name"]}?',
            'answer': f'{app_info["name"]} is {app_info["description"]}.'
        })
        
        # Collect documentation files
        for doc_file in app_info['docs']:
            doc_path = app_path / doc_file
            if doc_path.exists():
                content = self.read_file_safe(doc_path)
                if content:
                    sections = self.extract_markdown_sections(content)
                    
                    for section in sections:
                        title = section['title']
                        content_text = section['content']
                        
                        if len(content_text) > 50:  # Only include substantial sections
                            # Create Q&A pairs
                            qa_pairs.append({
                                'question': f'Tell me about {app_info["name"]} {title.lower()}',
                                'answer': f'{app_info["name"]} {title}:\n\n{content_text[:2000]}'  # Limit length
                            })
                            
                            # Extract key features/concepts
                            if 'feature' in title.lower() or 'overview' in title.lower():
                                qa_pairs.append({
                                    'question': f'What are the key features of {app_info["name"]}?',
                                    'answer': content_text[:1500]
                                })
        
        # Collect README summary
        readme_path = app_path / 'README.md'
        if readme_path.exists():
            readme_content = self.read_file_safe(readme_path)
            if readme_content:
                # Extract first few paragraphs as summary
                paragraphs = [p.strip() for p in readme_content.split('\n\n') if p.strip()][:5]
                summary = '\n\n'.join(paragraphs)
                
                qa_pairs.append({
                    'question': f'Give me an overview of {app_info["name"]}',
                    'answer': summary[:2000]
                })
        
        logger.info(f"Collected {len(qa_pairs)} Q&A pairs for {app_info['name']}")
        return qa_pairs
    
    def collect_code_context(self, app_key: str) -> List[Dict[str, str]]:
        """
        Collect code context (API endpoints, models, etc.).
        
        Returns:
            List of Q&A pairs about code structure
        """
        app_info = self.applications[app_key]
        app_path = app_info['path']
        
        qa_pairs = []
        
        # Collect API endpoints if available
        if app_key == 'canopy':
            api_path = app_path / 'backend' / 'api'
            if api_path.exists():
                api_files = list(api_path.glob('*.py'))
                if api_files:
                    endpoints_info = []
                    for api_file in api_files[:5]:  # Limit to first 5 files
                        content = self.read_file_safe(api_file)
                        # Extract route definitions
                        routes = re.findall(r'@app\.(get|post|put|delete)\(["\']([^"\']+)["\']', content)
                        if routes:
                            endpoints_info.append(f"{api_file.name}: {', '.join([f'{m[0].upper()} {m[1]}' for m in routes])}")
                    
                    if endpoints_info:
                        qa_pairs.append({
                            'question': f'What API endpoints does {app_info["name"]} have?',
                            'answer': f'{app_info["name"]} provides the following API endpoints:\n\n' + '\n'.join(endpoints_info)
                        })
        
        elif app_key == 'swimTO':
            api_path = app_path / 'apps' / 'api'
            if api_path.exists():
                qa_pairs.append({
                    'question': f'What is the API structure of {app_info["name"]}?',
                    'answer': f'{app_info["name"]} has a FastAPI backend located in apps/api/ that provides endpoints for pool schedules and facilities data.'
                })
        
        return qa_pairs
    
    def collect_all(self, output_dir: str) -> Dict[str, List[Dict[str, str]]]:
        """
        Collect all application data.
        
        Args:
            output_dir: Directory to save collected data
            
        Returns:
            Dictionary mapping app keys to Q&A pairs
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        all_data = {}
        
        for app_key in self.applications.keys():
            logger.info(f"\nCollecting data for {app_key}...")
            
            # Collect documentation
            doc_qa = self.collect_application_docs(app_key)
            
            # Collect code context
            code_qa = self.collect_code_context(app_key)
            
            # Combine
            all_qa = doc_qa + code_qa
            all_data[app_key] = all_qa
            
            # Save individual app data
            app_output_file = output_path / f'{app_key}_qa.json'
            with open(app_output_file, 'w', encoding='utf-8') as f:
                json.dump(all_qa, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(all_qa)} Q&A pairs to {app_output_file}")
        
        # Save combined data
        combined_file = output_path / 'applications_qa.json'
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\nSaved combined data to {combined_file}")
        
        # Create formatted text file for training
        text_file = output_path / 'applications_training.txt'
        with open(text_file, 'w', encoding='utf-8') as f:
            for app_key, qa_pairs in all_data.items():
                app_name = self.applications[app_key]['name']
                f.write(f"\n\n=== {app_name} ===\n\n")
                for qa in qa_pairs:
                    f.write(f"Q: {qa['question']}\n\n")
                    f.write(f"A: {qa['answer']}\n\n")
                    f.write("-" * 80 + "\n\n")
        
        logger.info(f"Saved formatted training text to {text_file}")
        
        return all_data


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect application data for Nima training')
    parser.add_argument('--workspace-root', type=str, default='../..',
                       help='Root directory containing applications (default: ../..)')
    parser.add_argument('--output-dir', type=str, 
                       default=os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'applications'),
                       help='Output directory for collected data')
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = Path(__file__).parent
    workspace_root = Path(args.workspace_root).resolve()
    if not workspace_root.is_absolute():
        workspace_root = script_dir / workspace_root
    
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir
    
    logger.info(f"Workspace root: {workspace_root}")
    logger.info(f"Output directory: {output_dir}")
    
    # Collect data
    collector = ApplicationDataCollector(str(workspace_root))
    all_data = collector.collect_all(str(output_dir))
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("DATA COLLECTION SUMMARY")
    logger.info("=" * 80)
    total_qa = sum(len(qa_list) for qa_list in all_data.values())
    logger.info(f"Total Q&A pairs collected: {total_qa}")
    for app_key, qa_list in all_data.items():
        logger.info(f"  {app_key}: {len(qa_list)} pairs")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()




