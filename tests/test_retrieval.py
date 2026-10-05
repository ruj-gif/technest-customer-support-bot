"""Basic tests for the TechNest document loader and local retrieval."""

import tempfile
import unittest
from pathlib import Path

from src.retrieval import load_documents, retrieve


PROJECT_DATA = Path(__file__).resolve().parents[1] / "data"


class RetrievalTests(unittest.TestCase):
    def test_loads_markdown_and_product_catalogue(self) -> None:
        documents = load_documents(PROJECT_DATA)
        sources = {document["source"] for document in documents}

        self.assertIn("faq.md", sources)
        self.assertIn("return_policy.md", sources)
        self.assertIn("products.csv", sources)
        self.assertTrue(any("TechNest AeroBook 14" in item["text"] for item in documents))
        self.assertTrue(any("Order tracking" in item["text"] for item in documents))

    def test_retrieval_returns_ranked_results_with_sources_and_scores(self) -> None:
        results = retrieve("How long does delivery take?", data_dir=PROJECT_DATA)

        self.assertGreaterEqual(len(results), 1)
        self.assertLessEqual(len(results), 3)
        self.assertEqual(results[0]["source"], "faq.md")
        self.assertGreater(results[0]["score"], 0)
        self.assertIn("delivery", results[0]["text"].lower())

    def test_empty_query_returns_no_results(self) -> None:
        self.assertEqual(retrieve("   ", data_dir=PROJECT_DATA), [])

    def test_missing_files_are_handled_gracefully(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            self.assertEqual(load_documents(temporary_directory), [])
            self.assertEqual(retrieve("store hours", data_dir=temporary_directory), [])

    def test_product_rows_keep_all_catalogue_fields(self) -> None:
        products = [
            item
            for item in load_documents(PROJECT_DATA)
            if item["source"] == "products.csv"
        ]

        self.assertEqual(len(products), 9)
        product = next(item for item in products if item["product_id"] == "TN-KEY-109")
        self.assertEqual(product["product_name"], "TechNest Compact Keyboard")
        self.assertEqual(product["category"], "Accessories")
        self.assertEqual(product["price_inr"], "1790")
        self.assertEqual(product["stock_status"], "In stock")
        self.assertEqual(product["warranty"], "12-month manufacturer warranty")
        self.assertIn("quiet keys", product["description"])

    def test_broad_product_question_lists_the_catalogue(self) -> None:
        results = retrieve("What products do you sell?", data_dir=PROJECT_DATA)

        self.assertEqual(len(results), 9)
        self.assertTrue(all(item["source"] == "products.csv" for item in results))
        product_names = {item["product_name"] for item in results}
        self.assertIn("TechNest AeroBook 14", product_names)
        self.assertIn("TechNest Compact Keyboard", product_names)

    def test_headphone_query_returns_matching_product(self) -> None:
        results = retrieve("Do you have headphones?", data_dir=PROJECT_DATA)

        self.assertTrue(results)
        self.assertEqual(results[0]["product_id"], "TN-HDP-104")
        self.assertEqual(results[0]["source"], "products.csv")

    def test_keyboard_price_query_returns_correct_product_and_price(self) -> None:
        results = retrieve("What is the price of the keyboard?", data_dir=PROJECT_DATA)

        self.assertTrue(results)
        self.assertEqual(results[0]["product_id"], "TN-KEY-109")
        self.assertEqual(results[0]["price_inr"], "1790")
        self.assertEqual(results[0]["source"], "products.csv")

    def test_product_term_does_not_override_faq_warranty_retrieval(self) -> None:
        results = retrieve("What is the warranty on laptops?", data_dir=PROJECT_DATA)

        self.assertTrue(results)
        self.assertEqual(results[0]["source"], "faq.md")


if __name__ == "__main__":
    unittest.main()
