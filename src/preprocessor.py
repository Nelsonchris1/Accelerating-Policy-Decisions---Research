"""
Document Preprocessing Module for Climate Policy Documents

This module provides comprehensive preprocessing capabilities for climate policy documents,
including text cleaning, feature extraction, and document annotation.
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import hashlib

# NLP and text processing imports (will be available when requirements are installed)
try:
    from textstat import flesch_reading_ease, flesch_kincaid_grade
    from langdetect import detect
    import spacy
    from transformers import pipeline
except ImportError:
    print("Some dependencies not installed. Run: pip install -r requirements.txt")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentPreprocessor:
    """
    Comprehensive document preprocessing for climate policy texts.
    
    Features:
    - Text cleaning and normalization
    - Language detection and translation
    - Named entity recognition
    - Policy-specific feature extraction
    - Document annotation and labeling
    """
    
    def __init__(self, language_model: str = "en_core_web_sm"):
        """
        Initialize the document preprocessor.
        
        Args:
            language_model: Spacy language model to use
        """
        self.language_model = language_model
        
        # Initialize NLP pipeline
        try:
            self.nlp = spacy.load(language_model)
        except OSError:
            logger.warning(f"Language model {language_model} not found. Using basic processing.")
            self.nlp = None
            
        # Initialize sentiment analyzer
        try:
            self.sentiment_analyzer = pipeline("sentiment-analysis", 
                                             model="cardiffnlp/twitter-roberta-base-sentiment-latest")
        except Exception as e:
            logger.warning(f"Could not load sentiment analyzer: {e}")
            self.sentiment_analyzer = None
            
        # Climate policy specific terms and patterns
        self.climate_terms = {
            "mitigation": ["carbon reduction", "emissions reduction", "decarbonization", 
                          "renewable energy", "energy efficiency", "carbon capture"],
            "adaptation": ["climate resilience", "adaptation measures", "climate risk", 
                          "vulnerability assessment", "adaptive capacity"],
            "finance": ["climate finance", "green bonds", "carbon pricing", 
                       "loss and damage", "climate fund"],
            "governance": ["climate governance", "policy framework", "international cooperation",
                          "transparency", "monitoring reporting verification"],
            "technology": ["clean technology", "technology transfer", "innovation",
                          "research and development", "deployment"]
        }
        
        # Document types based on research
        self.document_types = [
            "policy_document", "scientific_report", "implementation_guideline",
            "economic_analysis", "regional_assessment", "technical_report",
            "conference_proceedings", "working_paper", "policy_brief"
        ]
        
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
            
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep climate-relevant symbols
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\%\°\(\)]', ' ', text)
        
        # Fix common OCR errors in policy documents
        ocr_fixes = {
            r'\bl\b': 'I',  # Common OCR error
            r'\b0\b': 'O',  # Zero instead of O
            r'(\d+)\s*°\s*C': r'\1°C',  # Temperature formatting
            r'CO\s*2': 'CO2',  # CO2 formatting
            r'(\d+)\s*%': r'\1%'  # Percentage formatting
        }
        
        for pattern, replacement in ocr_fixes.items():
            text = re.sub(pattern, replacement, text)
            
        return text
    
    def extract_metadata(self, text: str, title: str = "") -> Dict[str, Any]:
        """
        Extract metadata from document text.
        
        Args:
            text: Document text
            title: Document title
            
        Returns:
            Dictionary of extracted metadata
        """
        metadata = {
            "char_count": len(text),
            "word_count": len(text.split()),
            "sentence_count": len(re.findall(r'[.!?]+', text)),
            "paragraph_count": len([p for p in text.split('\n\n') if p.strip()]),
            "title": title
        }
        
        # Language detection
        try:
            metadata["language"] = detect(text)
        except Exception:
            metadata["language"] = "unknown"
            
        # Readability scores
        try:
            metadata["flesch_reading_ease"] = flesch_reading_ease(text)
            metadata["flesch_kincaid_grade"] = flesch_kincaid_grade(text)
        except Exception:
            metadata["flesch_reading_ease"] = None
            metadata["flesch_kincaid_grade"] = None
            
        # Document hash for deduplication (using secure SHA256)
        metadata["content_hash"] = hashlib.sha256(text.encode()).hexdigest()
        
        return metadata
    
    def extract_climate_features(self, text: str) -> Dict[str, Any]:
        """
        Extract climate policy specific features.
        
        Args:
            text: Document text
            
        Returns:
            Dictionary of climate-specific features
        """
        text_lower = text.lower()
        features = {}
        
        # Count occurrences of climate terms by category
        for category, terms in self.climate_terms.items():
            count = sum(text_lower.count(term) for term in terms)
            features[f"{category}_mentions"] = count
            
        # Detect policy instruments mentioned
        policy_instruments = [
            "carbon tax", "emissions trading", "renewable energy standard",
            "energy efficiency standard", "feed-in tariff", "carbon offset",
            "green building standard", "fuel economy standard"
        ]
        
        features["policy_instruments"] = [
            instrument for instrument in policy_instruments 
            if instrument in text_lower
        ]
        
        # Extract numeric targets and dates
        targets = re.findall(r'(\d+(?:\.\d+)?)\s*%?\s*(?:by|until|before)\s*(\d{4})', text)
        features["numeric_targets"] = targets
        
        # Geographic scope detection
        regions = ["global", "national", "regional", "local", "international"]
        features["geographic_scope"] = [
            region for region in regions if region in text_lower
        ]
        
        # Urgency indicators
        urgency_terms = ["urgent", "immediate", "critical", "emergency", "rapid"]
        features["urgency_score"] = sum(text_lower.count(term) for term in urgency_terms)
        
        return features
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text.
        
        Args:
            text: Text to process
            
        Returns:
            Dictionary of entity types and their values
        """
        entities = {
            "organizations": [],
            "countries": [],
            "dates": [],
            "percentages": [],
            "technologies": [],
            "policies": []
        }
        
        if self.nlp is None:
            # Fallback to regex-based extraction
            entities["dates"] = re.findall(r'\b\d{4}\b', text)
            entities["percentages"] = re.findall(r'\d+(?:\.\d+)?%', text)
            return entities
            
        # Use spaCy for entity extraction
        doc = self.nlp(text)
        
        for ent in doc.ents:
            if ent.label_ == "ORG":
                entities["organizations"].append(ent.text)
            elif ent.label_ == "GPE":
                entities["countries"].append(ent.text)
            elif ent.label_ == "DATE":
                entities["dates"].append(ent.text)
                
        # Custom entity extraction for climate-specific terms
        climate_entities = {
            "technologies": ["solar", "wind", "nuclear", "hydroelectric", "geothermal",
                           "carbon capture", "storage", "electric vehicle"],
            "policies": ["paris agreement", "kyoto protocol", "cop", "unfccc",
                        "green deal", "carbon tax", "cap and trade"]
        }
        
        text_lower = text.lower()
        for entity_type, terms in climate_entities.items():
            entities[entity_type] = [
                term for term in terms if term in text_lower
            ]
            
        return entities
    
    def classify_document_type(self, text: str, title: str = "") -> Dict[str, float]:
        """
        Classify document type based on content.
        
        Args:
            text: Document text
            title: Document title
            
        Returns:
            Dictionary of document type probabilities
        """
        # Simple rule-based classification
        scores = {}
        full_text = (title + " " + text).lower()
        
        # Define patterns for each document type
        type_patterns = {
            "policy_document": ["policy", "regulation", "law", "act", "decree"],
            "scientific_report": ["research", "study", "analysis", "findings", "methodology"],
            "implementation_guideline": ["guideline", "implementation", "procedure", "instruction"],
            "economic_analysis": ["economic", "cost", "benefit", "financial", "investment"],
            "regional_assessment": ["regional", "country", "national", "local", "assessment"],
            "technical_report": ["technical", "technology", "engineering", "specification"],
            "conference_proceedings": ["conference", "proceedings", "workshop", "symposium"],
            "working_paper": ["working paper", "draft", "preliminary", "discussion"],
            "policy_brief": ["brief", "summary", "overview", "key points"]
        }
        
        for doc_type, patterns in type_patterns.items():
            score = sum(full_text.count(pattern) for pattern in patterns)
            scores[doc_type] = score / len(patterns)  # Normalize by pattern count
            
        # Normalize scores to probabilities
        total_score = sum(scores.values())
        if total_score > 0:
            scores = {k: v/total_score for k, v in scores.items()}
        else:
            # Default uniform distribution
            scores = {k: 1/len(type_patterns) for k in type_patterns}
            
        return scores
    
    def annotate_document(self, text: str, title: str = "", 
                         additional_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create comprehensive document annotation.
        
        Args:
            text: Document text
            title: Document title
            additional_metadata: Additional metadata to include
            
        Returns:
            Complete document annotation
        """
        logger.info(f"Annotating document: {title[:50]}...")
        
        # Clean text
        clean_text = self.clean_text(text)
        
        # Extract all features
        annotation = {
            "original_text": text,
            "cleaned_text": clean_text,
            "metadata": self.extract_metadata(clean_text, title),
            "climate_features": self.extract_climate_features(clean_text),
            "entities": self.extract_entities(clean_text),
            "document_type_scores": self.classify_document_type(clean_text, title)
        }
        
        # Add sentiment if available
        if self.sentiment_analyzer:
            try:
                # Analyze sentiment of first 512 tokens (model limit)
                text_sample = ' '.join(clean_text.split()[:512])
                sentiment = self.sentiment_analyzer(text_sample)[0]
                annotation["sentiment"] = sentiment
            except Exception as e:
                logger.warning(f"Sentiment analysis failed: {e}")
                annotation["sentiment"] = None
        
        # Add additional metadata
        if additional_metadata:
            annotation["additional_metadata"] = additional_metadata
            
        # Determine primary document type
        doc_types = annotation["document_type_scores"]
        annotation["primary_document_type"] = max(doc_types, key=doc_types.get)
        
        logger.info(f"Annotation complete. Primary type: {annotation['primary_document_type']}")
        return annotation
    
    def batch_process(self, documents: List[Dict[str, str]], 
                     output_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Process multiple documents in batch.
        
        Args:
            documents: List of documents with 'text' and 'title' keys
            output_path: Optional path to save results
            
        Returns:
            List of annotated documents
        """
        logger.info(f"Processing {len(documents)} documents in batch")
        
        annotated_docs = []
        for i, doc in enumerate(documents):
            logger.info(f"Processing document {i+1}/{len(documents)}")
            
            annotation = self.annotate_document(
                doc.get("text", ""),
                doc.get("title", f"Document {i+1}"),
                doc.get("metadata", {})
            )
            
            # Add original document ID if available
            if "id" in doc:
                annotation["document_id"] = doc["id"]
                
            annotated_docs.append(annotation)
            
        # Save results if path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(annotated_docs, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to: {output_path}")
            
        return annotated_docs
    
    def get_processing_stats(self, annotated_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate statistics from processed documents.
        
        Args:
            annotated_docs: List of annotated documents
            
        Returns:
            Processing statistics
        """
        if not annotated_docs:
            return {}
            
        stats = {
            "total_documents": len(annotated_docs),
            "languages": {},
            "document_types": {},
            "average_word_count": 0,
            "total_climate_mentions": 0
        }
        
        word_counts = []
        climate_mentions = []
        
        for doc in annotated_docs:
            # Language distribution
            lang = doc["metadata"].get("language", "unknown")
            stats["languages"][lang] = stats["languages"].get(lang, 0) + 1
            
            # Document type distribution
            doc_type = doc.get("primary_document_type", "unknown")
            stats["document_types"][doc_type] = stats["document_types"].get(doc_type, 0) + 1
            
            # Word count
            word_count = doc["metadata"].get("word_count", 0)
            word_counts.append(word_count)
            
            # Climate mentions
            climate_features = doc.get("climate_features", {})
            total_mentions = sum(v for k, v in climate_features.items() 
                               if k.endswith("_mentions") and isinstance(v, int))
            climate_mentions.append(total_mentions)
            
        stats["average_word_count"] = sum(word_counts) / len(word_counts) if word_counts else 0
        stats["total_climate_mentions"] = sum(climate_mentions)
        stats["average_climate_mentions"] = sum(climate_mentions) / len(climate_mentions) if climate_mentions else 0
        
        return stats


# Utility functions
def load_document_from_file(file_path: str) -> str:
    """Load document text from file."""
    path = Path(file_path)
    if path.suffix.lower() == '.txt':
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    elif path.suffix.lower() == '.pdf':
        # Would use PyPDF2 or pdfplumber here
        return f"PDF content from {file_path} would be extracted here"
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


if __name__ == "__main__":
    # Example usage
    preprocessor = DocumentPreprocessor()
    
    # Sample climate policy text
    sample_text = """
    The Paris Agreement on climate change represents a landmark international accord 
    that aims to limit global warming to well below 2°C above pre-industrial levels.
    Countries must implement nationally determined contributions (NDCs) that include
    significant emissions reductions by 2030. Key policy instruments include carbon 
    pricing mechanisms, renewable energy standards, and energy efficiency measures.
    """
    
    # Annotate document
    annotation = preprocessor.annotate_document(sample_text, "Sample Policy Document")
    
    print("Document annotation complete:")
    print(f"Primary type: {annotation['primary_document_type']}")
    print(f"Word count: {annotation['metadata']['word_count']}")
    print(f"Climate mentions: {sum(annotation['climate_features'][k] for k in annotation['climate_features'] if k.endswith('_mentions'))}")