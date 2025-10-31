#!/usr/bin/env python3
"""
Data collection script for climate policy datasets.

This script downloads and organizes climate policy documents and datasets
from various sources identified through research.
"""

import os
import sys
import json
import time
import requests
from pathlib import Path
from typing import Dict, List, Any
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from data_loader import DataLoader
except ImportError:
    print("Could not import DataLoader. Please install dependencies.")


def collect_cop29_data(output_dir: Path, limit: int = 100):
    """Collect COP29 documents and metadata."""
    print(f"Collecting COP29 data (limit: {limit})")
    
    # Create data loader
    loader = DataLoader(cache_dir=str(output_dir / "cache"))
    
    # Load COP29 documents
    documents = loader.load_cop29_documents(limit=limit)
    
    # Save to output directory
    output_file = output_dir / "cop29_documents.parquet"
    documents.to_parquet(output_file)
    
    print(f"Saved {len(documents)} COP29 documents to {output_file}")
    
    # Save metadata
    metadata = {
        "collection_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "UNFCCC COP29 Documents",
        "document_count": len(documents),
        "document_types": documents['type'].value_counts().to_dict()
    }
    
    metadata_file = output_dir / "cop29_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return documents


def collect_climate_datasets(output_dir: Path):
    """Collect various climate datasets."""
    print("Collecting climate datasets")
    
    loader = DataLoader(cache_dir=str(output_dir / "cache"))
    
    # Load climate datasets
    datasets = loader.load_climate_datasets()
    
    # Save each dataset
    for name, data in datasets.items():
        output_file = output_dir / f"climate_{name}.parquet"
        data.to_parquet(output_file)
        print(f"Saved {name} dataset to {output_file}")
    
    # Save combined metadata
    metadata = {
        "collection_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "datasets": {
            name: {
                "shape": list(data.shape),
                "columns": list(data.columns)
            }
            for name, data in datasets.items()
        }
    }
    
    metadata_file = output_dir / "climate_datasets_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)


def collect_research_papers(output_dir: Path, limit: int = 50):
    """Collect research paper metadata."""
    print(f"Collecting research papers (limit: {limit})")
    
    loader = DataLoader(cache_dir=str(output_dir / "cache"))
    
    # Load research papers
    papers = loader.load_research_papers(limit=limit)
    
    # Save papers
    output_file = output_dir / "research_papers.parquet"
    papers.to_parquet(output_file)
    
    print(f"Saved {len(papers)} research papers to {output_file}")
    
    # Save metadata
    metadata = {
        "collection_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "paper_count": len(papers),
        "year_distribution": papers.groupby('year').size().to_dict(),
        "journal_distribution": papers.groupby('journal').size().to_dict()
    }
    
    metadata_file = output_dir / "research_papers_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)


def download_sample_pdfs(output_dir: Path):
    """Download sample climate policy PDFs."""
    print("Creating sample PDF references")
    
    # Sample URLs based on research (these would be real URLs in practice)
    sample_documents = [
        {
            "title": "COP29 Sustainability Report",
            "url": "https://unfccc.int/sites/default/files/resource/COP29_Sustainability_Report.pdf",
            "type": "sustainability_report",
            "source": "UNFCCC"
        },
        {
            "title": "Global Climate Finance Landscape 2024",
            "url": "https://www.climatepolicyinitiative.org/wp-content/uploads/2024/10/Global-Landscape-of-Climate-Finance-2024.pdf",
            "type": "financial_analysis",
            "source": "Climate Policy Initiative"
        },
        {  
            "title": "Machine Learning Map of Climate Policy Literature",
            "url": "https://www.nature.com/articles/s44168-024-00196-0.pdf",
            "type": "research_paper",
            "source": "Nature"
        }
    ]
    
    # Save reference list (actual downloading would require proper handling)
    pdf_dir = output_dir / "pdfs"
    pdf_dir.mkdir(exist_ok=True)
    
    reference_file = pdf_dir / "document_references.json"
    with open(reference_file, 'w') as f:
        json.dump(sample_documents, f, indent=2)
    
    print(f"Saved PDF references to {reference_file}")
    print("Note: Actual PDF downloading requires proper authentication and rate limiting")


def create_dataset_index(output_dir: Path):
    """Create an index of all collected datasets."""
    print("Creating dataset index")
    
    index = {
        "creation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "datasets": []
    }
    
    # Scan output directory for data files
    for file_path in output_dir.glob("*.parquet"):
        try:
            import pandas as pd
            df = pd.read_parquet(file_path)
            
            dataset_info = {
                "name": file_path.stem,
                "file": file_path.name,
                "format": "parquet",
                "size_mb": file_path.stat().st_size / (1024 * 1024),
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": list(df.columns)
            }
            
            index["datasets"].append(dataset_info)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Save index
    index_file = output_dir / "dataset_index.json"
    with open(index_file, 'w') as f:
        json.dump(index, f, indent=2)
    
    print(f"Dataset index saved to {index_file}")
    print(f"Indexed {len(index['datasets'])} datasets")


def main():
    """Main data collection function."""
    parser = argparse.ArgumentParser(description="Collect climate policy datasets")
    
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw",
        help="Output directory for downloaded data"
    )
    
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["cop29", "climate-datasets", "research-papers", "pdfs", "all"],
        default=["all"],
        help="Data sources to collect"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Limit for document collections"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Climate Policy Data Collection Tool")
    print(f"Output directory: {output_dir}")
    print(f"Sources: {args.sources}")
    print("-" * 50)
    
    # Collect data from specified sources
    if "all" in args.sources or "cop29" in args.sources:
        try:
            collect_cop29_data(output_dir, args.limit)
        except Exception as e:
            print(f"Error collecting COP29 data: {e}")
    
    if "all" in args.sources or "climate-datasets" in args.sources:
        try:
            collect_climate_datasets(output_dir)
        except Exception as e:
            print(f"Error collecting climate datasets: {e}")
    
    if "all" in args.sources or "research-papers" in args.sources:
        try:
            collect_research_papers(output_dir, args.limit)
        except Exception as e:
            print(f"Error collecting research papers: {e}")
    
    if "all" in args.sources or "pdfs" in args.sources:
        try:
            download_sample_pdfs(output_dir)
        except Exception as e:
            print(f"Error with PDF references: {e}")
    
    # Create dataset index
    try:
        create_dataset_index(output_dir)
    except Exception as e:
        print(f"Error creating dataset index: {e}")
    
    print("\nData collection complete!")
    print(f"All data saved to: {output_dir}")


if __name__ == "__main__":
    main()