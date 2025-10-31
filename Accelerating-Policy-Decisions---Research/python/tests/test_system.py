"""
Test suite for the climate policy document processing system.
"""

import unittest
import sys
from pathlib import Path
import json
import tempfile
import shutil

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from data_loader import DataLoader, validate_document_data
    from preprocessor import DocumentPreprocessor
    from evaluator import PolicyEvaluator, create_sample_evaluation_data
    import pandas as pd
except ImportError as e:
    print(f"Import error: {e}")
    print("Some tests may be skipped due to missing dependencies")


class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.loader = DataLoader(cache_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_load_cop29_documents(self):
        """Test loading COP29 documents."""
        try:
            docs = self.loader.load_cop29_documents(limit=5)
            self.assertIsInstance(docs, pd.DataFrame)
            self.assertEqual(len(docs), 5)
            self.assertTrue(validate_document_data(docs))
        except ImportError:
            self.skipTest("pandas not available")
    
    def test_load_climate_datasets(self):
        """Test loading climate datasets."""
        try:
            datasets = self.loader.load_climate_datasets()
            self.assertIsInstance(datasets, dict)
            self.assertIn("co2_emissions", datasets)
            self.assertIn("policy_instruments", datasets)
        except ImportError:
            self.skipTest("pandas not available")
    
    def test_load_research_papers(self):
        """Test loading research papers."""
        try:
            papers = self.loader.load_research_papers(limit=3)
            self.assertIsInstance(papers, pd.DataFrame)
            self.assertEqual(len(papers), 3)
            required_columns = ["id", "title", "authors", "year"]
            for col in required_columns:
                self.assertIn(col, papers.columns)
        except ImportError:
            self.skipTest("pandas not available")
    
    def test_cache_functionality(self):
        """Test caching functionality."""
        test_data = {"test": "data", "timestamp": "2024-01-01"}
        
        # Save to cache
        self.loader.save_to_cache(test_data, "test_cache")
        
        # Load from cache
        loaded_data = self.loader.load_from_cache("test_cache")
        self.assertEqual(loaded_data, test_data)
    
    def test_get_dataset_info(self):
        """Test dataset information retrieval."""
        info = self.loader.get_dataset_info()
        self.assertIsInstance(info, dict)
        self.assertIn("cop29_documents", info)
        self.assertIn("climate_datasets", info)


class TestDocumentPreprocessor(unittest.TestCase):
    """Test cases for DocumentPreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = DocumentPreprocessor()
        self.sample_text = """
        The Paris Agreement on climate change represents a landmark international accord 
        that aims to limit global warming to well below 2°C above pre-industrial levels.
        Countries must implement nationally determined contributions (NDCs) that include
        significant emissions reductions by 2030. Key policy instruments include carbon 
        pricing mechanisms, renewable energy standards, and energy efficiency measures.
        Urgent action is required to address the climate emergency.
        """
    
    def test_clean_text(self):
        """Test text cleaning functionality."""
        dirty_text = "This  is   a\n\n  test\t\ttext   with   extra  spaces."
        cleaned = self.preprocessor.clean_text(dirty_text)
        self.assertNotIn("  ", cleaned)  # No double spaces
        self.assertNotIn("\n\n", cleaned)  # No double newlines
    
    def test_extract_metadata(self):
        """Test metadata extraction."""
        metadata = self.preprocessor.extract_metadata(self.sample_text, "Test Document")
        
        # Check required fields
        self.assertIn("char_count", metadata)
        self.assertIn("word_count", metadata)
        self.assertIn("sentence_count", metadata)
        self.assertIn("title", metadata)
        self.assertIn("content_hash", metadata)
        
        # Check values are reasonable
        self.assertGreater(metadata["word_count"], 0)
        self.assertGreater(metadata["char_count"], 0)
        self.assertEqual(metadata["title"], "Test Document")
    
    def test_extract_climate_features(self):
        """Test climate-specific feature extraction."""
        features = self.preprocessor.extract_climate_features(self.sample_text)
        
        # Check climate term categories
        self.assertIn("mitigation_mentions", features)
        self.assertIn("adaptation_mentions", features)
        self.assertIn("finance_mentions", features)
        
        # Check policy instruments detection
        self.assertIn("policy_instruments", features)
        self.assertIsInstance(features["policy_instruments"], list)
        
        # Check urgency score
        self.assertIn("urgency_score", features)
        self.assertGreater(features["urgency_score"], 0)  # Should detect "urgent"
    
    def test_classify_document_type(self):
        """Test document type classification."""
        scores = self.preprocessor.classify_document_type(self.sample_text, "Policy Framework")
        
        self.assertIsInstance(scores, dict)
        self.assertGreater(len(scores), 0)
        
        # Check scores sum to approximately 1
        total_score = sum(scores.values())
        self.assertAlmostEqual(total_score, 1.0, places=2)
    
    def test_annotate_document(self):
        """Test complete document annotation."""
        annotation = self.preprocessor.annotate_document(
            self.sample_text, 
            "Test Policy Document"
        )
        
        # Check annotation structure
        required_keys = [
            "original_text", "cleaned_text", "metadata", 
            "climate_features", "entities", "document_type_scores",
            "primary_document_type"
        ]
        
        for key in required_keys:
            self.assertIn(key, annotation)
    
    def test_batch_process(self):
        """Test batch processing of documents."""
        documents = [
            {
                "id": "doc1",
                "title": "Climate Policy 1",
                "text": "This is a test climate policy document about carbon emissions."
            },
            {
                "id": "doc2", 
                "title": "Climate Policy 2",
                "text": "This document discusses renewable energy and adaptation measures."
            }
        ]
        
        # Process batch
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "batch_results.json"
            results = self.preprocessor.batch_process(documents, str(output_path))
            
            self.assertEqual(len(results), 2)
            self.assertTrue(output_path.exists())
            
            # Check first result
            self.assertIn("document_id", results[0])
            self.assertEqual(results[0]["document_id"], "doc1")


class TestPolicyEvaluator(unittest.TestCase):
    """Test cases for PolicyEvaluator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = PolicyEvaluator()
    
    def test_evaluate_retrieval_performance(self):
        """Test retrieval performance evaluation."""
        retrieved_docs = [
            {"id": "doc1", "score": 0.9},
            {"id": "doc2", "score": 0.8},
            {"id": "doc3", "score": 0.7},
            {"id": "doc4", "score": 0.6}
        ]
        relevant_docs = ["doc1", "doc3", "doc5"]
        
        metrics = self.evaluator.evaluate_retrieval_performance(
            retrieved_docs, relevant_docs, k_values=[2, 4]
        )
        
        # Check metrics exist
        self.assertIn("precision_at_2", metrics)
        self.assertIn("precision_at_4", metrics)
        self.assertIn("ndcg_at_2", metrics)
        self.assertIn("mrr", metrics)
        self.assertIn("map", metrics)
        
        # Check value ranges
        for metric, value in metrics.items():
            self.assertGreaterEqual(value, 0)
            self.assertLessEqual(value, 1)
    
    def test_evaluate_response_quality(self):
        """Test response quality evaluation."""
        generated = [
            "Climate policy requires immediate carbon emission reductions.",
            "Renewable energy is essential for sustainable development."
        ]
        reference = [
            "Immediate reductions in carbon emissions are needed for climate policy.",
            "Sustainable development requires investment in renewable energy sources."
        ]
        
        metrics = self.evaluator.evaluate_response_quality(generated, reference)
        
        # Check metrics structure
        self.assertIsInstance(metrics, dict)
        self.assertIn("avg_coherence_score", metrics)
        self.assertIn("avg_relevance_score", metrics)
    
    def test_evaluate_policy_effectiveness(self):
        """Test policy effectiveness evaluation."""
        policy_docs = [
            {
                "id": "policy1",
                "metadata": {"flesch_reading_ease": 60, "word_count": 500},
                "climate_features": {
                    "policy_instruments": ["carbon tax", "renewable standard"],
                    "numeric_targets": [("50", "2030")],
                    "urgency_score": 3,
                    "geographic_scope": ["national", "regional"]
                },
                "entities": {"dates": ["2030", "2025"]},
                "cleaned_text": "monitor and verify emissions track progress"
            }
        ]
        
        evaluation = self.evaluator.evaluate_policy_effectiveness(policy_docs)
        
        # Check evaluation structure
        self.assertIn("overall_score", evaluation)
        self.assertIn("criteria_scores", evaluation)
        self.assertIn("policy_scores", evaluation)
        self.assertIn("recommendations", evaluation)
        
        # Check overall score range
        self.assertGreaterEqual(evaluation["overall_score"], 0)
        self.assertLessEqual(evaluation["overall_score"], 1)
    
    def test_evaluate_user_satisfaction(self):
        """Test user satisfaction evaluation."""
        feedback = [
            {
                "overall_rating": 9,
                "accuracy": 8,
                "relevance": 9,
                "interpretability": 8,
                "response_time_rating": 9,
                "trust_rating": 8
            },
            {
                "overall_rating": 8,
                "accuracy": 9,
                "relevance": 8,
                "interpretability": 9,
                "response_time_rating": 8,
                "trust_rating": 9
            }
        ]
        
        metrics = self.evaluator.evaluate_user_satisfaction(feedback)
        
        # Check metrics
        self.assertIn("avg_overall_satisfaction", metrics)
        self.assertIn("composite_satisfaction", metrics)
        
        # Check reasonable values
        self.assertGreater(metrics["composite_satisfaction"], 7)  # Should be high
    
    def test_comprehensive_evaluation(self):
        """Test comprehensive evaluation pipeline."""
        evaluation_data = create_sample_evaluation_data()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "evaluation_report.json"
            
            report = self.evaluator.run_comprehensive_evaluation(
                evaluation_data, str(output_path)
            )
            
            # Check report structure
            self.assertIn("evaluation_timestamp", report)
            self.assertIn("benchmarks", report)
            self.assertIn("results", report)
            self.assertIn("overall_assessment", report)
            
            # Check file was created
            self.assertTrue(output_path.exists())


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def test_full_pipeline(self):
        """Test the complete processing pipeline."""
        try:
            # Load data
            loader = DataLoader()
            docs = loader.load_cop29_documents(limit=2)
            
            # Convert to document format
            documents = []
            for _, row in docs.iterrows():
                documents.append({
                    "id": row["id"],
                    "title": row["title"],
                    "text": row.get("content_preview", "Sample policy text about climate change.")
                })
            
            # Preprocess documents
            preprocessor = DocumentPreprocessor()
            annotated_docs = preprocessor.batch_process(documents)
            
            # Basic checks
            self.assertEqual(len(annotated_docs), 2)
            self.assertIn("primary_document_type", annotated_docs[0])
            
            # Generate stats
            stats = preprocessor.get_processing_stats(annotated_docs)
            self.assertIn("total_documents", stats)
            self.assertEqual(stats["total_documents"], 2)
            
        except ImportError:
            self.skipTest("pandas not available for integration test")


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestDataLoader,
        TestDocumentPreprocessor, 
        TestPolicyEvaluator,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Running Climate Policy Processing System Tests")
    print("=" * 60)
    
    success = run_tests()
    
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)