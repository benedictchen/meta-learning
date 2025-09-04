#!/usr/bin/env python3
"""
Demonstration of the Modular Qualitative Reasoning Core

This script showcases the key features of the integrated modular architecture,
including different use cases and factory functions.
"""

def demo_basic_usage():
    """Demonstrate basic usage of the modular core"""
    
    print("🧠 Basic Usage Demo")
    print("=" * 30)
    
    from qualitative_reasoning import QualitativeReasoner, QualitativeValue, QualitativeDirection
    
    # Create a basic reasoner
    reasoner = QualitativeReasoner("Heat Transfer System")
    
    # Add quantities with different initial states
    reasoner.add_quantity("temperature", QualitativeValue.ZERO, QualitativeDirection.STEADY, 
                         landmarks=[0, 100], units="°C")
    reasoner.add_quantity("heat_flow", QualitativeValue.POSITIVE_SMALL, QualitativeDirection.INCREASING,
                         units="W")
    reasoner.add_quantity("thermal_energy", QualitativeValue.POSITIVE_SMALL, QualitativeDirection.STEADY,
                         units="J")
    
    # Add process with causal influences
    reasoner.add_process("heating",
                        preconditions=["heat_source_active"],
                        quantity_conditions=["heat_flow > 0"],
                        influences=["I+(temperature)", "I+(thermal_energy)"])
    
    # Add system constraints
    reasoner.add_constraint("temperature >= 0")
    reasoner.add_constraint("thermal_energy >= 0")
    
    # Run simulation steps
    print("\n🚀 Running simulation...")
    reasoner.run_simulation("initial_heating")
    reasoner.run_simulation("continued_heating")
    
    # Generate explanation
    print("\n🔍 Analyzing temperature behavior...")
    explanation = reasoner.explain_quantity("temperature")
    print(f"Primary causes: {explanation.primary_causes}")
    print(f"Confidence: {explanation.confidence:.2f}")
    
    # Predict future states
    print("\n🔮 Predicting future states...")
    predictions = reasoner.predict_future(2)
    for i, pred in enumerate(predictions, 1):
        print(f"  Step {i}: Temperature = {pred.quantities['temperature'].magnitude.value}")
    
    print("\n✓ Basic demo completed!\n")


def demo_factory_functions():
    """Demonstrate different factory functions for various use cases"""
    
    print("🏭 Factory Functions Demo")
    print("=" * 30)
    
    from qualitative_reasoning import (
        create_educational_reasoner,
        create_research_reasoner,
        create_production_reasoner,
        create_demo_reasoner
    )
    
    # Educational use case
    print("\n🎓 Educational Reasoner:")
    edu_reasoner = create_educational_reasoner("Physics 101: Thermodynamics")
    edu_reasoner.add_quantity("student_temperature", landmarks=[0, 100])
    edu_reasoner.add_process("learning", ["curiosity_present"], [], ["I+(student_temperature)"])
    edu_reasoner.run_simulation("learning_phase")
    
    # Research use case  
    print("\n🔬 Research Reasoner:")
    research_reasoner = create_research_reasoner("Advanced Heat Transfer", enable_predictions=True)
    research_reasoner.add_quantity("complex_temperature")
    research_reasoner.add_quantity("entropy")
    research_reasoner.add_process("entropy_increase", [], ["temperature > 0"], ["I+(entropy)"])
    
    # Production use case
    print("\n🏭 Production Reasoner:")
    prod_reasoner = create_production_reasoner("Industrial Heating", security_level="high")
    prod_reasoner.add_quantity("furnace_temp")
    prod_reasoner.add_constraint("furnace_temp <= 1000")  # Safety constraint
    
    # Demo use case
    print("\n🎬 Demo Reasoner:")
    demo_reasoner = create_demo_reasoner("Conference Presentation")
    demo_reasoner.add_quantity("audience_interest")
    demo_reasoner.add_process("engagement", ["presenter_active"], [], ["I+(audience_interest)"])
    
    print("✓ All factory functions demonstrated!\n")


def demo_security_features():
    """Demonstrate security features and constraint evaluation methods"""
    
    print("🔒 Security Features Demo")
    print("=" * 30)
    
    from qualitative_reasoning import QualitativeReasoner, ConstraintEvaluationMethod
    
    # Create reasoner with different security levels
    reasoner = QualitativeReasoner("Security Test System")
    reasoner.add_quantity("secure_temp")
    
    # Test AST Safe method (default and recommended)
    print("\n🛡️  Testing AST Safe method...")
    reasoner.configure_security(ConstraintEvaluationMethod.AST_SAFE, strict_mode=False)
    reasoner.add_constraint("secure_temp > 0")  # Safe constraint
    reasoner.run_simulation("ast_safe_test")
    
    # Test Hybrid method for maximum compatibility
    print("\n🔄 Testing Hybrid method...")
    reasoner.configure_security(ConstraintEvaluationMethod.HYBRID)
    reasoner.run_simulation("hybrid_test")
    
    # Show security status
    security_status = reasoner.get_constraint_security_status()
    print(f"\n📊 Security Status:")
    print(f"  Method: {security_status['evaluation_method']}")
    print(f"  Strict mode: {security_status['strict_mode']}")
    print(f"  Function calls allowed: {security_status['allow_function_calls']}")
    
    print("✓ Security features demonstrated!\n")


def demo_analysis_and_visualization():
    """Demonstrate advanced analysis and visualization capabilities"""
    
    print("🎨 Analysis & Visualization Demo")
    print("=" * 40)
    
    from qualitative_reasoning import create_research_reasoner
    
    # Create research-grade reasoner
    reasoner = create_research_reasoner("Complex System Analysis")
    
    # Build a more complex system
    reasoner.add_quantity("input_flow", landmarks=[0, 10, 50])
    reasoner.add_quantity("tank_level", landmarks=[0, 25, 50, 75, 100])
    reasoner.add_quantity("output_flow", landmarks=[0, 5, 20])
    reasoner.add_quantity("pressure", landmarks=[0, 1, 5, 10])
    
    # Add interconnected processes
    reasoner.add_process("filling",
                        preconditions=["valve_open"],
                        quantity_conditions=["input_flow > 0"],
                        influences=["I+(tank_level)"])
    
    reasoner.add_process("pressure_buildup",
                        preconditions=["tank_not_empty"],
                        quantity_conditions=["tank_level > 0"],
                        influences=["I+(pressure)"])
    
    reasoner.add_process("draining",
                        preconditions=["outlet_open"],
                        quantity_conditions=["pressure > 0"],
                        influences=["I+(output_flow)", "I-(tank_level)"])
    
    # Add system constraints
    reasoner.add_constraint("tank_level >= 0")
    reasoner.add_constraint("pressure >= 0")
    
    # Run multiple simulation steps
    print("\n🚀 Running complex simulation...")
    for step in range(1, 4):
        reasoner.run_simulation(f"complex_step_{step}")
    
    # Generate comprehensive analysis
    print("\n📊 Generating comprehensive report...")
    report = reasoner.generate_report("text", include_history=True, include_predictions=False)
    
    # Show system status
    status = reasoner.get_system_status()
    print(f"\n📈 System Status Summary:")
    print(f"  Quantities: {status['quantities']['count']}")
    print(f"  Active processes: {status['processes']['active']}/{status['processes']['total']}")
    print(f"  Activity rate: {status['processes']['activity_rate']:.1%}")
    print(f"  Simulation steps: {status['history']['steps']}")
    
    # Export data in different formats
    print("\n💾 Exporting system data...")
    json_data = reasoner.export_system_state(format_type="json")
    print(f"  JSON export: {len(json_data)} characters")
    
    csv_data = reasoner.export_system_state(format_type="csv")  
    print(f"  CSV export: {len(csv_data.splitlines())} lines")
    
    print("✓ Analysis and visualization demo completed!\n")


def main():
    """Run all demonstrations"""
    
    print("🌟 Modular Qualitative Reasoning Core Demo")
    print("=" * 50)
    print()
    print("This demonstration shows the key features of the modular")
    print("qualitative reasoning system, including:")
    print("• Integrated modular architecture")
    print("• Factory functions for different use cases")
    print("• Security-first constraint evaluation")
    print("• Rich analysis and visualization capabilities")
    print("• Full backward compatibility")
    print()
    
    # Run all demonstrations
    demo_basic_usage()
    demo_factory_functions()
    demo_security_features()
    demo_analysis_and_visualization()
    
    print("🎉 All demonstrations completed successfully!")
    print()
    print("The modular qualitative reasoning core provides:")
    print("✓ Clean separation of concerns through mixins")
    print("✓ Security-first design with no eval() vulnerabilities")
    print("✓ Rich analysis and explanation capabilities") 
    print("✓ Multiple visualization and export formats")
    print("✓ Factory functions for common use cases")
    print("✓ Full backward compatibility with original API")
    print()
    print("Ready for educational, research, and production use!")


if __name__ == "__main__":
    main()