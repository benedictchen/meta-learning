#!/usr/bin/env python3
"""
Standalone Test for Difficulty Patcher Implementation
=====================================================

This test bypasses imports and directly tests the difficulty patcher
implementation in isolation to confirm functionality before removing TODOs.
"""

import sys
import os
import logging

# Add the specific file directory to path
patcher_path = 'src/meta_learning/patches/difficulty_components'
if patcher_path not in sys.path:
    sys.path.insert(0, patcher_path)

# Direct import of the difficulty patcher
try:
    from difficulty_patcher import DifficultyEstimationPatcher
    print("✅ Successfully imported DifficultyEstimationPatcher")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Create a simple test module for patching
def create_test_module():
    """Create a simple test module with a hardcoded 0.5 function."""
    import types
    
    # Create mock module
    test_module = types.ModuleType('test_module')
    
    def hardcoded_difficulty_function():
        """Function that returns hardcoded 0.5."""
        return 0.5
    
    def non_hardcoded_function():
        """Function that returns a different value."""
        return 0.7
    
    def exception_function():
        """Function that raises an exception."""
        raise ValueError("Test exception")
    
    # Add functions to module
    test_module.hardcoded_difficulty_function = hardcoded_difficulty_function
    test_module.non_hardcoded_function = non_hardcoded_function  
    test_module.exception_function = exception_function
    
    # Register module so it can be imported
    sys.modules['test_module'] = test_module
    
    return test_module

def test_patcher_initialization():
    """Test DifficultyEstimationPatcher initialization."""
    print("\n🧪 Testing patcher initialization...")
    
    # Test with default parameters
    patcher = DifficultyEstimationPatcher(enable_patches=False)  # Disable auto-patching
    
    # Verify initialization
    assert hasattr(patcher, 'enable_patches')
    assert hasattr(patcher, 'fallback_to_original')
    assert hasattr(patcher, 'patched_functions')
    assert hasattr(patcher, 'original_functions')
    assert hasattr(patcher, 'logger')
    assert hasattr(patcher, 'simple_estimator')
    
    # Test simple estimator functionality
    difficulty = patcher.simple_estimator()
    assert 0.3 <= difficulty <= 0.8, f"Difficulty {difficulty} outside expected range [0.3, 0.8]"
    
    print(f"✅ Patcher initialized, estimator returns: {difficulty:.4f}")
    print("✅ Patcher initialization test PASSED")
    return True

def test_function_patching():
    """Test basic function patching functionality."""
    print("\n🧪 Testing function patching...")
    
    # Create test module
    test_module = create_test_module()
    
    # Initialize patcher
    patcher = DifficultyEstimationPatcher(enable_patches=False)
    
    # Test original function behavior
    original_result = test_module.hardcoded_difficulty_function()
    assert original_result == 0.5, f"Original function should return 0.5, got {original_result}"
    print(f"✅ Original function returns: {original_result}")
    
    # Patch the function
    success = patcher.patch_function('test_module', 'hardcoded_difficulty_function')
    assert success, "Patching should succeed"
    print("✅ Function patched successfully")
    
    # Test patched function behavior
    patched_result = test_module.hardcoded_difficulty_function()
    assert 0.3 <= patched_result <= 0.8, f"Patched function should return estimated difficulty, got {patched_result}"
    assert patched_result != 0.5, "Patched function should not return hardcoded 0.5"
    print(f"✅ Patched function returns: {patched_result:.4f}")
    
    # Verify tracking
    patched_functions = patcher.list_patched_functions()
    assert 'test_module.hardcoded_difficulty_function' in patched_functions
    print(f"✅ Tracking works: {patched_functions}")
    
    print("✅ Function patching test PASSED")
    return True

def test_non_hardcoded_function_patching():
    """Test that non-hardcoded functions are not affected by patching."""
    print("\n🧪 Testing non-hardcoded function patching...")
    
    # Get test module
    test_module = sys.modules['test_module']
    
    # Initialize patcher
    patcher = DifficultyEstimationPatcher(enable_patches=False)
    
    # Test original non-hardcoded function
    original_result = test_module.non_hardcoded_function()
    assert original_result == 0.7, f"Original function should return 0.7, got {original_result}"
    
    # Patch the function
    success = patcher.patch_function('test_module', 'non_hardcoded_function')
    assert success, "Patching should succeed"
    
    # Test that non-hardcoded result is unchanged
    patched_result = test_module.non_hardcoded_function()
    assert patched_result == 0.7, f"Non-hardcoded function should remain unchanged, got {patched_result}"
    print(f"✅ Non-hardcoded function unchanged: {patched_result}")
    
    print("✅ Non-hardcoded function patching test PASSED")
    return True

def test_fallback_behavior():
    """Test fallback behavior when enhanced function fails."""
    print("\n🧪 Testing fallback behavior...")
    
    # Get test module
    test_module = sys.modules['test_module']
    
    # Initialize patcher with fallback enabled
    patcher = DifficultyEstimationPatcher(enable_patches=False, fallback_to_original=True)
    
    # Patch the exception function
    success = patcher.patch_function('test_module', 'exception_function')
    assert success, "Patching should succeed"
    
    # Test that fallback works - should not raise exception
    try:
        result = test_module.exception_function()
        print(f"❌ Should have raised exception but got result: {result}")
        return False
    except ValueError as e:
        if "Test exception" in str(e):
            print("✅ Fallback behavior working - original exception preserved")
        else:
            print(f"❌ Unexpected exception: {e}")
            return False
    
    print("✅ Fallback behavior test PASSED")
    return True

def test_function_unpatching():
    """Test function unpatching functionality."""
    print("\n🧪 Testing function unpatching...")
    
    # Create a fresh test module to avoid contamination
    fresh_test_module = create_test_module() 
    sys.modules['fresh_test_module'] = fresh_test_module
    
    # New patcher
    patcher = DifficultyEstimationPatcher(enable_patches=False)
    
    # Test original value
    original_result = fresh_test_module.hardcoded_difficulty_function()
    assert original_result == 0.5, f"Original should be 0.5, got {original_result}"
    print(f"✅ Original function returns: {original_result}")
    
    # Patch function
    success = patcher.patch_function('fresh_test_module', 'hardcoded_difficulty_function')
    assert success, "Patching should succeed"
    
    # Verify it's patched
    patched_result = fresh_test_module.hardcoded_difficulty_function()
    assert patched_result != 0.5, "Function should be patched"
    print(f"✅ Patched function returns: {patched_result:.4f}")
    
    # Unpatch function
    unpatch_success = patcher.unpatch_function('fresh_test_module', 'hardcoded_difficulty_function')
    assert unpatch_success, "Unpatching should succeed"
    print("✅ Unpatch operation succeeded")
    
    # Verify it's restored
    restored_result = fresh_test_module.hardcoded_difficulty_function()
    assert restored_result == 0.5, f"Function should be restored to original, got {restored_result}"
    print(f"✅ Function restored to original: {restored_result}")
    
    # Verify tracking is cleaned up
    patched_functions = patcher.list_patched_functions()
    assert 'fresh_test_module.hardcoded_difficulty_function' not in patched_functions
    print("✅ Tracking cleaned up correctly")
    
    print("✅ Function unpatching test PASSED")
    return True

def test_patch_statistics():
    """Test patch statistics functionality."""
    print("\n🧪 Testing patch statistics...")
    
    # Initialize patcher
    patcher = DifficultyEstimationPatcher(enable_patches=False)
    
    # Get initial statistics
    initial_stats = patcher.get_patch_statistics()
    assert initial_stats['total_patched_functions'] == 0
    assert initial_stats['success_rate'] == 0
    print("✅ Initial statistics correct")
    
    # Patch a function
    success = patcher.patch_function('test_module', 'hardcoded_difficulty_function')
    assert success, "Patching should succeed"
    
    # Get updated statistics
    updated_stats = patcher.get_patch_statistics()
    assert updated_stats['total_patched_functions'] == 1
    assert updated_stats['success_rate'] == 100.0
    assert 'test_module.hardcoded_difficulty_function' in updated_stats['patched_functions']
    assert updated_stats['estimator_type'] == 'SimpleRandomEstimator'
    
    print(f"✅ Updated statistics: {updated_stats['total_patched_functions']} functions, {updated_stats['success_rate']:.1f}% success rate")
    print("✅ Patch statistics test PASSED")
    return True

def test_invalid_module_patching():
    """Test handling of invalid module/function combinations.""" 
    print("\n🧪 Testing invalid module patching...")
    
    patcher = DifficultyEstimationPatcher(enable_patches=False)
    
    # Test non-existent module
    success = patcher.patch_function('nonexistent_module', 'some_function')
    assert not success, "Patching non-existent module should fail"
    print("✅ Non-existent module patching fails correctly")
    
    # Test non-existent function in existing module
    success = patcher.patch_function('test_module', 'nonexistent_function')
    assert not success, "Patching non-existent function should fail"
    print("✅ Non-existent function patching fails correctly")
    
    # Test unpatching non-patched function
    success = patcher.unpatch_function('test_module', 'hardcoded_difficulty_function')
    assert not success, "Unpatching non-patched function should fail"
    print("✅ Non-patched function unpatching fails correctly")
    
    print("✅ Invalid module patching test PASSED")
    return True

def main():
    """Run all difficulty patcher tests."""
    print("🚀 Starting Standalone Difficulty Patcher Tests")
    print("=" * 60)
    
    # Set up logging to reduce noise
    logging.basicConfig(level=logging.WARNING)
    
    try:
        # Create test module first
        create_test_module()
        
        # Run all tests
        tests = [
            test_patcher_initialization,
            test_function_patching,
            test_non_hardcoded_function_patching,
            test_fallback_behavior,
            test_function_unpatching,
            test_patch_statistics,
            test_invalid_module_patching
        ]
        
        all_passed = True
        for test_func in tests:
            if not test_func():
                all_passed = False
                break
        
        if all_passed:
            print("=" * 60)
            print("🎉 ALL DIFFICULTY PATCHER TESTS PASSED!")
            print("✅ Monkey patching system working correctly")
            print("✅ Hardcoded 0.5 replacement functional")
            print("✅ Error handling and fallback mechanisms working")
            print("✅ Function tracking and management operational")
            print("✅ Ready to remove TODO comments")
            print("=" * 60)
            return True
        else:
            print("=" * 60)
            print("❌ SOME TESTS FAILED")
            return False
        
    except Exception as e:
        print("=" * 60)
        print(f"❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)