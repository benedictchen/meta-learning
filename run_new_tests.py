#!/usr/bin/env python3
"""
Test Runner for New Functionality
=================================

This script runs all the comprehensive tests for the newly implemented functionality:
- FunctionalModule integration
- AdaptiveEpisodeSampler
- CurriculumSampler  
- OmniglotDataset
- Complete workflow integration
"""

import sys
import os
import subprocess
import time
from typing import List, Dict, Any


def run_test_file(test_file: str, verbose: bool = False) -> Dict[str, Any]:
    """Run a specific test file and return results."""
    print(f"🧪 Running {test_file}...")
    
    # Set up environment
    env = os.environ.copy()
    env['PYTHONPATH'] = 'src'
    
    # Run pytest on the specific file
    cmd = ['python', '-m', 'pytest', test_file, '-v' if verbose else '-q']
    
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd='/Users/benedictchen/work/research_papers/packages/meta_learning',
            env=env,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per test file
        )
        
        elapsed = time.time() - start_time
        
        success = result.returncode == 0
        
        if success:
            print(f"✅ {test_file} passed in {elapsed:.1f}s")
        else:
            print(f"❌ {test_file} failed in {elapsed:.1f}s")
            if verbose:
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
        
        return {
            'file': test_file,
            'success': success,
            'elapsed': elapsed,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
    except subprocess.TimeoutExpired:
        print(f"⏰ {test_file} timed out after 5 minutes")
        return {
            'file': test_file,
            'success': False,
            'elapsed': 300.0,
            'stdout': '',
            'stderr': 'Test timed out'
        }
    except Exception as e:
        print(f"💥 {test_file} crashed with error: {e}")
        return {
            'file': test_file,
            'success': False,
            'elapsed': time.time() - start_time,
            'stdout': '',
            'stderr': str(e)
        }


def run_all_new_tests(verbose: bool = False) -> Dict[str, Any]:
    """Run all new functionality tests."""
    print("=" * 70)
    print("🚀 Running Comprehensive Tests for New Functionality")
    print("=" * 70)
    
    # List of new test files to run
    test_files = [
        'tests/test_functional_module_integration.py',
        'tests/test_adaptive_episode_sampler.py', 
        'tests/test_curriculum_sampler.py',
        'tests/test_omniglot_dataset.py',
        'tests/test_integration_workflow.py',
        'tests/test_phase4_ml_enhancements.py',
        'tests/test_error_handling_comprehensive.py',
        'tests/test_evaluation_comprehensive.py',
        'tests/test_research_patches_determinism.py',
        'tests/test_performance_benchmarking.py'
    ]
    
    results = []
    total_start = time.time()
    
    # Run each test file
    for test_file in test_files:
        if os.path.exists(test_file):
            result = run_test_file(test_file, verbose)
            results.append(result)
        else:
            print(f"⚠️  Test file not found: {test_file}")
            results.append({
                'file': test_file,
                'success': False,
                'elapsed': 0.0,
                'stdout': '',
                'stderr': 'File not found'
            })
    
    total_elapsed = time.time() - total_start
    
    # Summarize results
    print("\n" + "=" * 70)
    print("📊 Test Results Summary")
    print("=" * 70)
    
    passed = sum(1 for r in results if r['success'])
    failed = len(results) - passed
    
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"⏱️  Total time: {total_elapsed:.1f}s")
    
    # Show details for failed tests
    if failed > 0:
        print("\n📝 Failed Test Details:")
        for result in results:
            if not result['success']:
                print(f"\n❌ {result['file']}:")
                if result['stderr']:
                    print(f"   Error: {result['stderr'][:200]}{'...' if len(result['stderr']) > 200 else ''}")
    
    # Overall result
    overall_success = failed == 0
    if overall_success:
        print("\n🎉 All new functionality tests passed!")
        print("✅ FunctionalModule integration working")
        print("✅ AdaptiveEpisodeSampler functional") 
        print("✅ CurriculumSampler operational")
        print("✅ OmniglotDataset implemented")
        print("✅ Complete workflow integration successful")
        print("✅ Phase 4 ML-powered enhancements functional")
        print("✅ Error handling and recovery mechanisms working")
        print("✅ Evaluation harness and metrics operational")
        print("✅ Research patches and determinism validated")
        print("✅ Performance benchmarking comprehensive")
    else:
        print(f"\n💥 {failed} test file(s) failed - see details above")
    
    return {
        'overall_success': overall_success,
        'passed': passed,
        'failed': failed,
        'total_elapsed': total_elapsed,
        'results': results
    }


def main():
    """Main test runner function."""
    # Parse command line arguments
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    
    if '--help' in sys.argv or '-h' in sys.argv:
        print("Usage: python run_new_tests.py [--verbose] [--help]")
        print()
        print("Run comprehensive tests for all newly implemented functionality:")
        print("  - FunctionalModule integration with MAML") 
        print("  - AdaptiveEpisodeSampler for difficulty adjustment")
        print("  - CurriculumSampler for progressive learning")
        print("  - OmniglotDataset implementation")
        print("  - Complete workflow integration tests")
        print("  - Phase 4 ML-powered enhancement components")
        print("  - Error handling and recovery mechanisms")
        print("  - Evaluation harness and metrics")
        print("  - Research patches and determinism hooks")
        print("  - Performance benchmarking and scalability")
        print()
        print("Options:")
        print("  --verbose, -v    Show detailed test output")
        print("  --help, -h       Show this help message")
        return
    
    # Check that we're in the right directory
    if not os.path.exists('src/meta_learning'):
        print("❌ Error: Please run this script from the meta_learning package root directory")
        print("   Expected to find: src/meta_learning/")
        sys.exit(1)
    
    # Run the tests
    results = run_all_new_tests(verbose=verbose)
    
    # Exit with appropriate code
    sys.exit(0 if results['overall_success'] else 1)


if __name__ == "__main__":
    main()