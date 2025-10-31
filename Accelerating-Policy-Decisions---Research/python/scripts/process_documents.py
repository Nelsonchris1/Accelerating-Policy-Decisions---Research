#!/usr/bin/env python3
"""
Main script for processing climate policy documents.

This script provides a command-line interface for loading, preprocessing,
and evaluating climate policy documents from various sources.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from data_loader import DataLoader
    from preprocessor import DocumentPreprocessor
    from evaluator import PolicyEvaluator
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install required dependencies: pip install -r requirements.txt")
    sys.exit(1)


def main():
    """Main function for CLI."""
    parser = argparse.ArgumentParser(
        description="Climate Policy Document Processing Tool"
    )
    
    parser.add_argument(
        "command",
        choices=["load", "preprocess", "evaluate", "full-pipeline"],
        help="Command to execute"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        help="Input file or directory path"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory path (default: output)"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Limit number of documents to process (default: 100)"
    )
    
    parser.add_argument(
        "--source",
        choices=["cop29", "climate-datasets", "research-papers", "local"],
        default="cop29",
        help="Data source (default: cop29)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Climate Policy Document Processing Tool")
    print(f"Command: {args.command}")
    print(f"Output directory: {output_path}")
    print("-" * 50)
    
    if args.command == "load":
        load_data(args, output_path)
    elif args.command == "preprocess":
        preprocess_documents(args, output_path)
    elif args.command == "evaluate":
        evaluate_system(args, output_path)
    elif args.command == "full-pipeline":
        run_full_pipeline(args, output_path)
    
    print("Processing complete!")


def load_data(args, output_path: Path):
    """Load data from specified source."""
    print(f"Loading data from source: {args.source}")
    
    loader = DataLoader(cache_dir=str(output_path / "cache"))
    
    if args.source == "cop29":
        data = loader.load_cop29_documents(limit=args.limit)
        filename = "cop29_documents.parquet"
    elif args.source == "climate-datasets":
        data = loader.load_climate_datasets()
        filename = "climate_datasets.json"
    elif args.source == "research-papers":
        data = loader.load_research_papers(limit=args.limit)
        filename = "research_papers.parquet"
    else:
        print(f"Unsupported source: {args.source}")
        return
    
    # Save data
    if hasattr(data, 'to_parquet'):  # DataFrame
        output_file = output_path / filename
        data.to_parquet(output_file)
        print(f"Data saved to: {output_file}")
        print(f"Shape: {data.shape}")
    else:  # Dictionary or other
        output_file = output_path / filename
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Data saved to: {output_file}")
        if isinstance(data, dict):
            print(f"Keys: {list(data.keys())}")


def preprocess_documents(args, output_path: Path):
    """Preprocess documents."""
    print("Preprocessing documents...")
    
    preprocessor = DocumentPreprocessor()
    
    # Load input data
    if args.input:
        input_path = Path(args.input)
        if input_path.suffix == '.parquet':
            import pandas as pd
            df = pd.read_parquet(input_path)
            documents = df.to_dict('records')
        elif input_path.suffix == '.json':
            with open(input_path, 'r') as f:
                documents = json.load(f)
        else:
            print(f"Unsupported input format: {input_path.suffix}")
            return
    else:
        # Use sample documents
        documents = [
            {
                "id": "sample_1",
                "title": "COP29 Climate Finance Agreement",
                "text": "The COP29 conference in Baku resulted in a historic agreement on climate finance, with developed countries committing to provide $300 billion annually by 2035 to support developing nations in their climate action efforts."
            },
            {
                "id": "sample_2", 
                "title": "National Climate Policy Framework",
                "text": "This framework establishes mandatory emissions reduction targets of 50% by 2030 relative to 2005 levels, with specific sector-by-sector implementation guidelines and monitoring mechanisms."
            }
        ]
    
    # Process documents
    output_file = output_path / "annotated_documents.json"
    annotated_docs = preprocessor.batch_process(documents, str(output_file))
    
    # Generate statistics
    stats = preprocessor.get_processing_stats(annotated_docs)
    stats_file = output_path / "processing_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Processed {len(annotated_docs)} documents")
    print(f"Results saved to: {output_file}")
    print(f"Statistics saved to: {stats_file}")


def evaluate_system(args, output_path: Path):
    """Evaluate system performance."""
    print("Evaluating system performance...")
    
    evaluator = PolicyEvaluator()
    
    # Create evaluation data (in practice, this would come from args.input)
    if args.input:
        with open(args.input, 'r') as f:
            evaluation_data = json.load(f)
    else:
        # Use sample evaluation data
        from evaluator import create_sample_evaluation_data
        evaluation_data = create_sample_evaluation_data()
    
    # Run evaluation
    output_file = output_path / "evaluation_report.json"
    report = evaluator.run_comprehensive_evaluation(
        evaluation_data, 
        str(output_file)
    )
    
    print(f"Evaluation complete. Report saved to: {output_file}")
    print(f"Overall assessment: {report['overall_assessment']['overall_rating']}")
    
    # Print key metrics
    if "retrieval" in report["results"]:
        retrieval = report["results"]["retrieval"]
        print(f"Precision@10: {retrieval.get('precision_at_10', 0):.3f}")
        print(f"NDCG@10: {retrieval.get('ndcg_at_10', 0):.3f}")


def run_full_pipeline(args, output_path: Path):
    """Run the complete processing pipeline."""
    print("Running full processing pipeline...")
    
    # Step 1: Load data
    print("\n1. Loading data...")
    load_args = argparse.Namespace(**vars(args))
    load_data(load_args, output_path)
    
    # Step 2: Preprocess documents
    print("\n2. Preprocessing documents...")
    preprocess_args = argparse.Namespace(**vars(args))
    # Set input to the loaded data
    if args.source == "cop29":
        preprocess_args.input = str(output_path / "cop29_documents.parquet")
    elif args.source == "research-papers":
        preprocess_args.input = str(output_path / "research_papers.parquet")
    
    preprocess_documents(preprocess_args, output_path)
    
    # Step 3: Evaluate system
    print("\n3. Evaluating system...")
    evaluate_args = argparse.Namespace(**vars(args))
    evaluate_system(evaluate_args, output_path)
    
    print("\nFull pipeline complete!")
    print(f"All outputs saved to: {output_path}")


if __name__ == "__main__":
    main()