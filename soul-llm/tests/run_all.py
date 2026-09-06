"""
SOUL-LLM Test Suite Runner & System Health Check
================================================
Runs all unit tests across the project:
1. Tokenizer (encode, decode, vocabulary)
2. Attention (causal masking, output shapes)
3. Model (forward pass, loss, generation)
4. Training (backpropagation, overfitting test)
5. API (health check, schema validation)
"""

import os
import sys
import unittest

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_all_tests():
    print("\n" + "=" * 60)
    print("      🧪 RUNNING COMPLETE SOUL-LLM TEST SUITE")
    print("=" * 60 + "\n")

    loader = unittest.TestLoader()
    start_dir = os.path.dirname(os.path.abspath(__file__))
    suite = loader.discover(start_dir, pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("🎉 ALL SOUL-LLM UNIT TESTS PASSED SUCCESSFULLY!")
        print(f"   Ran {result.testsRun} tests with 0 failures and 0 errors.")
    else:
        print("✖ SOME TESTS FAILED!")
        print(f"   Failures: {len(result.failures)}, Errors: {len(result.errors)}")
    print("=" * 60 + "\n")

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
