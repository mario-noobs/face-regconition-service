#!/usr/bin/env python3
"""
Test runner script for the face recognition service unit tests.
This script runs all unit tests and provides a comprehensive test report.
"""

import unittest
import sys
import os
import time
from io import StringIO

# Add the parent directory to the path so we can import test modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_all_tests():
    """Run all unit tests and return results"""
    # Create a test suite containing all test modules
    test_modules = [
        'tests.test_handlers',
        'tests.test_models',
        'tests.test_helpers',
        'tests.test_redis_db',
    ]
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Load tests from each module
    for module_name in test_modules:
        try:
            module_suite = loader.loadTestsFromName(module_name)
            suite.addTest(module_suite)
            print(f"✓ Loaded tests from {module_name}")
        except ImportError as e:
            print(f"✗ Failed to load tests from {module_name}: {e}")
            continue
    
    # Run the tests
    print(f"\n{'='*60}")
    print("RUNNING FACE RECOGNITION SERVICE UNIT TESTS")
    print(f"{'='*60}")
    
    # Capture test output
    stream = StringIO()
    runner = unittest.TextTestRunner(
        stream=stream,
        verbosity=2,
        buffer=True,
        failfast=False
    )
    
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    # Print results
    print(stream.getvalue())
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    
    # Print failure details
    if result.failures:
        print(f"\n{'='*60}")
        print("FAILURES")
        print(f"{'='*60}")
        for test, traceback in result.failures:
            print(f"\nFAILED: {test}")
            print(f"{'-'*40}")
            print(traceback)
    
    # Print error details  
    if result.errors:
        print(f"\n{'='*60}")
        print("ERRORS")
        print(f"{'='*60}")
        for test, traceback in result.errors:
            print(f"\nERROR: {test}")
            print(f"{'-'*40}")
            print(traceback)
    
    return result.wasSuccessful()

def run_specific_test_module(module_name):
    """Run tests from a specific module"""
    print(f"Running tests from {module_name}...")
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(f'tests.{module_name}')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def main():
    """Main function"""
    if len(sys.argv) > 1:
        # Run specific test module
        module_name = sys.argv[1]
        if not module_name.startswith('test_'):
            module_name = f'test_{module_name}'
        
        success = run_specific_test_module(module_name)
    else:
        # Run all tests
        success = run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()