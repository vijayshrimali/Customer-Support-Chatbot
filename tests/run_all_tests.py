"""
Master Test Runner
==================

Runs all test suites and generates comprehensive report.
"""

import os
import sys
from pathlib import Path
import time
from datetime import datetime

# Add parent directory to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# Import test modules
from test_knowledge_base import run_knowledge_base_tests
from test_graph_components import run_graph_component_tests
from test_end_to_end import run_end_to_end_tests


def print_header():
    """Print test suite header"""
    print("\n")
    print("╔" + "═"*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "  TechGear Electronics - Customer Support Chatbot".center(68) + "║")
    print("║" + "  COMPREHENSIVE TEST SUITE".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "═"*68 + "╝")
    print("\n")
    print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"Working Directory: {project_root}")
    print("\n")


def print_footer(total_time, all_results):
    """Print final summary"""
    print("\n")
    print("╔" + "═"*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "  FINAL TEST SUMMARY".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "═"*68 + "╝")
    print("\n")
    
    # Calculate totals
    total_tests = sum(r.testsRun for r in all_results)
    total_failures = sum(len(r.failures) for r in all_results)
    total_errors = sum(len(r.errors) for r in all_results)
    total_skipped = sum(len(r.skipped) for r in all_results)
    total_successes = total_tests - total_failures - total_errors - total_skipped
    
    print(f"Total Tests Run:        {total_tests}")
    print(f"Successful Tests:       {total_successes} ✅")
    print(f"Failed Tests:           {total_failures} ❌")
    print(f"Errors:                 {total_errors} ⚠️")
    print(f"Skipped Tests:          {total_skipped} ⏭️")
    print(f"\nTotal Execution Time:   {total_time:.2f} seconds")
    
    # Calculate success rate
    if total_tests > 0:
        success_rate = (total_successes / total_tests) * 100
        print(f"Success Rate:           {success_rate:.1f}%")
        
        if success_rate == 100:
            print("\n🎉 ALL TESTS PASSED! 🎉")
        elif success_rate >= 80:
            print("\n✅ Most tests passed - Good job!")
        elif success_rate >= 60:
            print("\n⚠️  Some tests failed - Review needed")
        else:
            print("\n❌ Many tests failed - Action required")
    
    print("\n" + "="*70)
    
    # Print test suite breakdown
    print("\nTEST SUITE BREAKDOWN:")
    print("-" * 70)
    
    suite_names = [
        "Knowledge Base & Embeddings",
        "LangGraph Components",
        "End-to-End Integration"
    ]
    
    for name, result in zip(suite_names, all_results):
        tests = result.testsRun
        failures = len(result.failures)
        errors = len(result.errors)
        skipped = len(result.skipped)
        successes = tests - failures - errors - skipped
        
        if tests > 0:
            rate = (successes / tests) * 100
            status = "✅" if rate == 100 else "⚠️" if rate >= 80 else "❌"
            print(f"{status} {name:.<45} {successes}/{tests} ({rate:.0f}%)")
    
    print("="*70 + "\n")


def check_environment():
    """Check if environment is properly configured"""
    print("🔍 Checking Test Environment...")
    print("-" * 70)
    
    issues = []
    
    # Check for .env file
    env_path = project_root / ".env"
    if not env_path.exists():
        issues.append("❌ .env file not found")
    else:
        print("✅ .env file found")
    
    # Check for knowledge base
    kb_path = project_root / "data" / "knowledge_base.txt"
    if not kb_path.exists():
        issues.append("❌ Knowledge base not found")
    else:
        print("✅ Knowledge base found")
    
    # Check for vector store
    db_path = project_root / "chroma_db"
    if not db_path.exists():
        issues.append("❌ ChromaDB vector store not found")
    else:
        print("✅ Vector store found")
    
    # Check for source files
    graph_path = project_root / "src" / "graph"
    if not graph_path.exists():
        issues.append("❌ Graph components not found")
    else:
        print("✅ Graph components found")
    
    print("-" * 70)
    
    if issues:
        print("\n⚠️  WARNINGS:")
        for issue in issues:
            print(f"  {issue}")
        print("\nSome tests may be skipped due to missing components.")
    else:
        print("\n✅ All components found - Ready to test!")
    
    print("\n")
    
    return len(issues) == 0


def run_all_tests():
    """Run all test suites"""
    print_header()
    
    # Check environment
    env_ok = check_environment()
    
    if not env_ok:
        response = input("Continue with tests anyway? (y/n): ")
        if response.lower() != 'y':
            print("Tests cancelled.")
            return
    
    # Start timer
    start_time = time.time()
    
    # Store all results
    all_results = []
    
    # Run test suites
    print("\n" + "="*70)
    print("STARTING TEST EXECUTION")
    print("="*70 + "\n")
    
    try:
        # Test 1: Knowledge Base & Embeddings
        print("\n📚 Phase 1/3: Testing Knowledge Base & Embeddings...")
        result1 = run_knowledge_base_tests()
        all_results.append(result1)
        
        # Test 2: Graph Components
        print("\n🔄 Phase 2/3: Testing LangGraph Components...")
        result2 = run_graph_component_tests()
        all_results.append(result2)
        
        # Test 3: End-to-End Integration
        print("\n🚀 Phase 3/3: Testing End-to-End Integration...")
        result3 = run_end_to_end_tests()
        all_results.append(result3)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user!")
        return
    except Exception as e:
        print(f"\n\n❌ Test execution failed: {e}")
        return
    
    # End timer
    end_time = time.time()
    total_time = end_time - start_time
    
    # Print final summary
    print_footer(total_time, all_results)
    
    # Save results to file
    save_test_report(all_results, total_time)


def save_test_report(all_results, total_time):
    """Save test results to a file"""
    report_path = project_root / "tests" / "test_report.txt"
    
    try:
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("TechGear Electronics - Customer Support Chatbot\n")
            f.write("COMPREHENSIVE TEST REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duration: {total_time:.2f} seconds\n\n")
            
            # Calculate totals
            total_tests = sum(r.testsRun for r in all_results)
            total_failures = sum(len(r.failures) for r in all_results)
            total_errors = sum(len(r.errors) for r in all_results)
            total_skipped = sum(len(r.skipped) for r in all_results)
            total_successes = total_tests - total_failures - total_errors - total_skipped
            
            f.write("SUMMARY:\n")
            f.write("-" * 70 + "\n")
            f.write(f"Total Tests:       {total_tests}\n")
            f.write(f"Passed:            {total_successes}\n")
            f.write(f"Failed:            {total_failures}\n")
            f.write(f"Errors:            {total_errors}\n")
            f.write(f"Skipped:           {total_skipped}\n")
            
            if total_tests > 0:
                success_rate = (total_successes / total_tests) * 100
                f.write(f"Success Rate:      {success_rate:.1f}%\n")
            
            f.write("\n" + "="*70 + "\n")
            
            # Suite breakdown
            suite_names = [
                "Knowledge Base & Embeddings",
                "LangGraph Components",
                "End-to-End Integration"
            ]
            
            f.write("\nTEST SUITE BREAKDOWN:\n")
            f.write("-" * 70 + "\n")
            
            for name, result in zip(suite_names, all_results):
                tests = result.testsRun
                failures = len(result.failures)
                errors = len(result.errors)
                skipped = len(result.skipped)
                successes = tests - failures - errors - skipped
                
                f.write(f"\n{name}:\n")
                f.write(f"  Total: {tests}, Passed: {successes}, Failed: {failures}, ")
                f.write(f"Errors: {errors}, Skipped: {skipped}\n")
                
                # Write failures if any
                if result.failures:
                    f.write("\n  Failures:\n")
                    for test, traceback in result.failures:
                        f.write(f"    - {test}\n")
                
                # Write errors if any
                if result.errors:
                    f.write("\n  Errors:\n")
                    for test, traceback in result.errors:
                        f.write(f"    - {test}\n")
            
            f.write("\n" + "="*70 + "\n")
        
        print(f"📄 Test report saved to: {report_path}")
        
    except Exception as e:
        print(f"⚠️  Could not save test report: {e}")


if __name__ == "__main__":
    run_all_tests()
