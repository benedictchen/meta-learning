#!/usr/bin/env python3
"""
🧠 Qualitative Reasoning Core Types - Demonstration Script
==========================================================

This script demonstrates the core types module extracted from the qualitative
reasoning system, showcasing the fundamental data structures and utilities
for qualitative physics reasoning.

Based on Forbus & de Kleer's foundational work in qualitative reasoning.
"""

from qualitative_reasoning import (
    # Core data types
    QualitativeValue, QualitativeDirection, 
    QualitativeQuantity, QualitativeState, QualitativeProcess,
    
    # Utility functions  
    create_quantity, compare_qualitative_values,
    qualitative_to_numeric, numeric_to_qualitative,
    validate_qualitative_state,
    
    # Type aliases for convenience
    QValue, QDirection, QQuantity, QState, QProcess
)


def demonstrate_qualitative_values():
    """Demonstrate qualitative value operations and comparisons."""
    print("🔢 Qualitative Value Demonstrations")
    print("-" * 40)
    
    # Create various qualitative values
    values = [
        QValue.NEGATIVE_LARGE,
        QValue.ZERO, 
        QValue.POSITIVE_SMALL,
        QValue.POSITIVE_INFINITY
    ]
    
    print("Qualitative value ordering:")
    for i, val in enumerate(values):
        print(f"  {i+1}. {val.value} (positive: {val.is_positive()}, negative: {val.is_negative()})")
    
    # Demonstrate comparisons
    print(f"\nComparisons:")
    print(f"  ZERO < POSITIVE_SMALL: {QValue.ZERO < QValue.POSITIVE_SMALL}")
    print(f"  NEGATIVE_LARGE < ZERO: {QValue.NEGATIVE_LARGE < QValue.ZERO}")
    
    # Using utility function
    result = compare_qualitative_values(QValue.POSITIVE_SMALL, QValue.POSITIVE_LARGE, "<")
    print(f"  Using utility: POSITIVE_SMALL < POSITIVE_LARGE: {result}")


def demonstrate_qualitative_directions():
    """Demonstrate qualitative direction operations."""
    print("\n📈 Qualitative Direction Demonstrations") 
    print("-" * 40)
    
    directions = [QDirection.INCREASING, QDirection.DECREASING, QDirection.STEADY]
    
    for direction in directions:
        print(f"  {direction.value} ({direction}): changing={direction.is_changing()}, "
              f"sign={direction.to_numeric_sign()}")
    
    # Demonstrate numeric conversion
    trend = 2.5
    qual_dir = QDirection.from_numeric_trend(trend)
    print(f"\nNumeric trend {trend} -> Qualitative direction: {qual_dir}")


def demonstrate_qualitative_quantities():
    """Demonstrate qualitative quantity creation and manipulation."""
    print("\n⚗️  Qualitative Quantity Demonstrations")
    print("-" * 40)
    
    # Create quantities using different methods
    temp = create_quantity(
        name="temperature",
        magnitude="pos_small", 
        direction="increasing",
        landmarks=[0.0, 25.0, 100.0],
        units="°C",
        description="System temperature"
    )
    
    pressure = QQuantity(
        name="pressure",
        magnitude=QValue.POSITIVE_LARGE,
        direction=QDirection.DECREASING,
        landmark_values=[0.0, 101325.0],  # 0 and 1 atmosphere
        units="Pa"
    )
    
    print(f"Temperature: {temp}")
    print(f"Pressure: {pressure}")
    
    # Demonstrate quantity methods
    print(f"\nQuantity analysis:")
    print(f"  Temperature is positive: {temp.is_positive()}")
    print(f"  Temperature is increasing: {temp.is_increasing()}")
    print(f"  Temperature is stable: {temp.is_stable()}")
    
    # Show magnitude transitions
    next_temp_mag = temp.transition_magnitude()
    print(f"  If temperature continues increasing: {next_temp_mag.value}")


def demonstrate_qualitative_states():
    """Demonstrate qualitative state management."""
    print("\n🏛️  Qualitative State Demonstrations")
    print("-" * 40)
    
    # Create a system state
    temp = create_quantity("temperature", "pos_small", "increasing", units="°C")
    pressure = create_quantity("pressure", "pos_large", "decreasing", units="Pa")
    flow = create_quantity("flow_rate", "zero", "steady", units="L/min")
    
    state = QState(
        time_point="t1",
        quantities={
            "temperature": temp,
            "pressure": pressure, 
            "flow_rate": flow
        },
        relationships={
            "temp_pressure_relation": "thermal_expansion",
            "pressure_flow_relation": "bernoulli_effect"
        }
    )
    
    print(f"System state: {state}")
    print(f"Quantities: {state.get_quantity_names()}")
    
    # Analyze state properties
    changing = [q.name for q in state.get_changing_quantities()]
    stable = [q.name for q in state.get_stable_quantities()]
    positive = [q.name for q in state.get_positive_quantities()]
    
    print(f"  Changing quantities: {changing}")
    print(f"  Stable quantities: {stable}")
    print(f"  Positive quantities: {positive}")
    
    # Validate state
    errors = validate_qualitative_state(state)
    if errors:
        print(f"  Validation errors: {errors}")
    else:
        print("  ✅ State validation: PASSED")


def demonstrate_qualitative_processes():
    """Demonstrate qualitative process definition."""
    print("\n⚙️  Qualitative Process Demonstrations")
    print("-" * 40)
    
    # Define a heating process
    heating = QProcess(
        name="heating",
        preconditions=["heat_source_active", "thermal_contact"],
        quantity_conditions=["temperature < max_temperature"],
        influences=["I+(temperature)", "I+(thermal_energy)"],
        description="Heat transfer from external source",
        priority=1
    )
    
    # Define a cooling process
    cooling = QProcess(
        name="cooling", 
        preconditions=["heat_sink_present"],
        quantity_conditions=["temperature > ambient_temperature"],
        influences=["I-(temperature)", "I-(thermal_energy)"],
        active=True,
        priority=2
    )
    
    print(f"Heating process: {heating}")
    print(f"Cooling process: {cooling}")
    
    # Analyze process properties
    heating_influences = heating.get_influenced_quantities()
    cooling_influences = cooling.get_influenced_quantities()
    
    print(f"\nProcess analysis:")
    print(f"  Heating influences: {heating_influences}")
    print(f"  Cooling influences: {cooling_influences}")
    print(f"  Heating effect on temperature: {heating.get_influence_type('temperature')}")
    print(f"  Does cooling affect temperature? {cooling.has_influence_on('temperature')}")


def demonstrate_utility_functions():
    """Demonstrate utility functions for qualitative reasoning."""
    print("\n🛠️  Utility Function Demonstrations")
    print("-" * 40)
    
    # Numeric <-> Qualitative conversions
    test_values = [0.0, 0.5, -2.0, 10.0, float('inf')]
    
    print("Numeric to qualitative conversions:")
    for val in test_values:
        qual_val = numeric_to_qualitative(val)
        numeric_back = qualitative_to_numeric(qual_val)
        print(f"  {val:8.1f} -> {qual_val.value:12} -> {numeric_back:8.1f}")
    
    # Landmark-based conversions
    landmarks = [0.0, 50.0, 100.0]
    test_temps = [-5.0, 25.0, 75.0, 150.0]
    
    print(f"\nLandmark-based conversion (landmarks: {landmarks}):")
    for temp in test_temps:
        qual_temp = numeric_to_qualitative(temp, landmarks)
        print(f"  {temp:6.1f}°C -> {qual_temp.value}")


def demonstrate_advanced_features():
    """Demonstrate advanced features and state comparisons."""
    print("\n🚀 Advanced Feature Demonstrations")
    print("-" * 40)
    
    # Create initial state
    initial_state = QState(
        time_point="t0",
        quantities={
            "temperature": create_quantity("temperature", "pos_small", "steady"),
            "pressure": create_quantity("pressure", "pos_large", "steady")
        }
    )
    
    # Create evolved state
    evolved_state = initial_state.copy()
    evolved_state.time_point = "t1"
    evolved_state.quantities["temperature"].direction = QDirection.INCREASING
    evolved_state.quantities["temperature"].magnitude = QValue.POSITIVE_LARGE
    evolved_state.add_relationship("heating_detected", "positive_temperature_trend")
    
    # Compare states
    differences = initial_state.compare_with(evolved_state)
    
    print("State evolution analysis:")
    print(f"  Initial state: {initial_state}")
    print(f"  Evolved state: {evolved_state}")
    print(f"  Differences: {differences}")
    
    # Demonstrate quantity copying and modification
    original_qty = create_quantity("volume", "pos_small", "increasing")
    modified_qty = original_qty.copy()
    modified_qty.direction = QDirection.DECREASING
    
    print(f"\nQuantity modification:")
    print(f"  Original: {original_qty}")
    print(f"  Modified: {modified_qty}")


def main():
    """Run all demonstrations."""
    print("🧠 Qualitative Reasoning Core Types - Comprehensive Demo")
    print("=" * 60)
    print("Based on Forbus & de Kleer's foundational qualitative reasoning work")
    print()
    
    demonstrate_qualitative_values()
    demonstrate_qualitative_directions()
    demonstrate_qualitative_quantities()
    demonstrate_qualitative_states()
    demonstrate_qualitative_processes()
    demonstrate_utility_functions()
    demonstrate_advanced_features()
    
    print("\n🎉 All demonstrations completed successfully!")
    print("✨ The core_types module provides a solid foundation for qualitative reasoning!")


if __name__ == "__main__":
    main()