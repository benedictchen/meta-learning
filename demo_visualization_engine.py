#!/usr/bin/env python3
"""
🎨 Visualization Engine Demonstration
=====================================

This script demonstrates the comprehensive visualization and reporting capabilities
of the extracted visualization engine module for qualitative reasoning systems.

The visualization engine provides the presentation layer that makes qualitative
reasoning results understandable through multiple output formats and visualization styles.

Author: Benedict Chen
"""

from qualitative_reasoning import QualitativeReasoner, QualitativeValue, QualitativeDirection
import json

def demo_basic_visualization():
    """Demonstrate basic system state visualization"""
    
    # Removed print spam: "...
    print("="*50)
    
    # Create a thermal system
    thermal_system = QualitativeReasoner("Thermal Control System")
    
    # Add quantities with different states
    thermal_system.add_quantity("temperature", QualitativeValue.POSITIVE_SMALL, 
                               QualitativeDirection.INCREASING, landmarks=[0.0, 100.0])
    thermal_system.add_quantity("pressure", QualitativeValue.POSITIVE_LARGE, 
                               QualitativeDirection.DECREASING)
    thermal_system.add_quantity("volume", QualitativeValue.POSITIVE_SMALL, 
                               QualitativeDirection.STEADY)
    thermal_system.add_quantity("entropy", QualitativeValue.POSITIVE_SMALL, 
                               QualitativeDirection.INCREASING)
    
    # Add processes
    thermal_system.add_process(
        "heating",
        preconditions=["heat_source_present"],
        quantity_conditions=["temperature > 0"],
        influences=["I+(temperature)", "I+(entropy)"]
    )
    
    thermal_system.add_process(
        "expansion",
        preconditions=["temperature_increasing"],
        quantity_conditions=["pressure > 0"],
        influences=["I+(volume)", "I-(pressure)"]
    )
    
    # Run simulation steps to create history
    for i in range(3):
        thermal_system.qualitative_simulation_step(f"t{i}")
    
    # Demonstrate basic visualization
    # Removed print spam: "\n...
    report = thermal_system.visualize_system_state(include_history=True)
    
    return thermal_system, report

def demo_advanced_reporting():
    """Demonstrate advanced reporting capabilities"""
    
    # Removed print spam: "\n...
    print("="*50)
    
    # Create a more complex system
    fluid_system = QualitativeReasoner("Fluid Dynamics System")
    
    # Add quantities
    fluid_system.add_quantity("flow_rate", QualitativeValue.POSITIVE_LARGE, 
                             QualitativeDirection.DECREASING)
    fluid_system.add_quantity("pressure_diff", QualitativeValue.POSITIVE_SMALL, 
                             QualitativeDirection.STEADY)
    fluid_system.add_quantity("viscosity", QualitativeValue.POSITIVE_SMALL, 
                             QualitativeDirection.INCREASING)
    
    # Add constraints
    fluid_system.add_constraint("flow_rate > 0")
    fluid_system.add_constraint("pressure_diff >= 0")
    
    # Add processes
    fluid_system.add_process(
        "flow_resistance",
        preconditions=["fluid_present"],
        quantity_conditions=["flow_rate > 0"],
        influences=["I-(flow_rate)", "I+(viscosity)"]
    )
    
    # Create history
    for i in range(4):
        fluid_system.qualitative_simulation_step(f"step_{i}")
    
    # Generate comprehensive report
    print("\n📋 Comprehensive Report:")
    comprehensive_report = fluid_system.generate_comprehensive_report(include_predictions=True)
    
    print(f"\nReport Overview:")
    print(f"- Title: {comprehensive_report.title}")
    print(f"- Sections: {list(comprehensive_report.sections.keys())}")
    print(f"- Data keys: {list(comprehensive_report.data.keys())}")
    
    return fluid_system, comprehensive_report

def demo_export_formats():
    """Demonstrate various export formats"""
    
    print("\n💾 Export Format Demonstration")  
    print("="*50)
    
    # Create a simple system
    export_system = QualitativeReasoner("Export Demo System")
    
    export_system.add_quantity("input", QualitativeValue.POSITIVE_SMALL, 
                              QualitativeDirection.INCREASING)
    export_system.add_quantity("output", QualitativeValue.POSITIVE_SMALL, 
                              QualitativeDirection.STEADY)
    export_system.add_quantity("buffer", QualitativeValue.ZERO, 
                              QualitativeDirection.INCREASING)
    
    # Add process
    export_system.add_process(
        "transfer",
        preconditions=["system_active"],
        quantity_conditions=["input > 0"],
        influences=["I+(buffer)", "I+(output)"]
    )
    
    # Run some steps
    for i in range(2):
        export_system.qualitative_simulation_step(f"export_t{i}")
    
    # Demonstrate export formats
    print("\n🔹 JSON Export:")
    json_export = export_system.export_data("json")
    json_data = json.loads(json_export)
    print(f"JSON contains: {list(json_data.keys())}")
    
    print("\n🔹 CSV Export:")
    csv_export = export_system.export_data("csv")
    print(f"CSV preview:\n{csv_export[:200]}...")
    
    print("\n🔹 Markdown Export:")
    markdown_export = export_system.export_data("markdown")
    lines = markdown_export.split('\n')
    print(f"Markdown has {len(lines)} lines, starts with: '{lines[0]}'")
    
    return export_system

def demo_history_rendering():
    """Demonstrate different history rendering modes"""
    
    # Removed print spam: "\n...
    print("="*50)
    
    # Create system with evolving quantities
    history_system = QualitativeReasoner("Dynamic Evolution System")
    
    history_system.add_quantity("energy", QualitativeValue.POSITIVE_SMALL, 
                               QualitativeDirection.INCREASING)
    history_system.add_quantity("stability", QualitativeValue.POSITIVE_LARGE, 
                               QualitativeDirection.DECREASING)
    
    history_system.add_process(
        "energy_transfer",
        preconditions=["active"],
        quantity_conditions=["energy >= 0"],
        influences=["I+(energy)", "I-(stability)"]
    )
    
    # Generate significant history
    for i in range(8):
        history_system.qualitative_simulation_step(f"history_t{i}")
    
    # Demonstrate different rendering modes
    # Removed print spam: "\n...
    timeline = history_system.render_state_history("timeline")
    print(timeline[:300] + "..." if len(timeline) > 300 else timeline)
    
    print("\n📋 Table Rendering:")
    table = history_system.render_state_history("table")
    print(table[:400] + "..." if len(table) > 400 else table)
    
    # Removed print spam: "\n...
    chart = history_system.render_state_history("chart")
    print(chart[:300] + "..." if len(chart) > 300 else chart)
    
    print("\n📄 Summary Rendering:")
    summary = history_system.render_state_history("summary")
    print(summary)
    
    return history_system

def demo_configuration_options():
    """Demonstrate visualization configuration options"""
    
    # Removed print spam: "\n...
    print("="*50)
    
    config_system = QualitativeReasoner("Configurable System")
    
    # Add some quantities
    config_system.add_quantity("signal", QualitativeValue.POSITIVE_SMALL, 
                              QualitativeDirection.STEADY)
    config_system.add_quantity("noise", QualitativeValue.NEGATIVE_SMALL, 
                              QualitativeDirection.INCREASING)
    
    # Configure visualization options
    print("\n🎨 Configuring visualization options...")
    config_system.configure_visualization(
        detail_level="comprehensive",
        max_history_items=15,
        include_metadata=True,
        include_explanations=True,
        export_format="json"
    )
    
    # Get visualization metrics
    metrics = config_system.get_visualization_metrics()
    print(f"\nVisualization Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Test visualization with new config
    config_system.qualitative_simulation_step("config_test")
    report = config_system.visualize_system_state()
    
    print(f"\nConfigured report has {len(report.sections)} sections")
    
    return config_system

def demo_analysis_integration():
    """Demonstrate integration with analysis engine"""
    
    print("\n🧠 Analysis Integration Demonstration")
    print("="*50)
    
    analysis_system = QualitativeReasoner("Integrated Analysis System")
    
    # Create a system with interesting relationships
    analysis_system.add_quantity("cause", QualitativeValue.POSITIVE_SMALL, 
                                QualitativeDirection.INCREASING)
    analysis_system.add_quantity("effect", QualitativeValue.ZERO, 
                                QualitativeDirection.STEADY)  
    analysis_system.add_quantity("mediator", QualitativeValue.POSITIVE_SMALL, 
                                QualitativeDirection.STEADY)
    
    analysis_system.add_process(
        "causation",
        preconditions=["active"],
        quantity_conditions=["cause > 0"],
        influences=["I+(effect)", "I+(mediator)"]
    )
    
    # Generate system evolution
    for i in range(5):
        analysis_system.qualitative_simulation_step(f"analysis_t{i}")
    
    # Generate comprehensive report with analysis
    # Removed print spam: "\n...
    integrated_report = analysis_system.generate_comprehensive_report()
    
    print(f"\nIntegrated Report Sections:")
    for section_name in integrated_report.sections.keys():
        print(f"  - {section_name}")
    
    # Show analysis section specifically
    if "Behavioral Patterns" in integrated_report.sections:
        # Removed print spam: f"\n...
        print(integrated_report.sections["Behavioral Patterns"][:300] + "...")
    
    return analysis_system

def main():
    """Main demonstration function"""
    
    print("🎨 Qualitative Reasoning Visualization Engine Demo")
    print("="*60)
    print("Demonstrating comprehensive visualization and reporting capabilities")
    print("extracted from the monolithic qualitative reasoning system.\n")
    
    try:
        # Run all demonstrations
        demo_basic_visualization()
        demo_advanced_reporting()
        demo_export_formats()
        demo_history_rendering()
        demo_configuration_options()
        demo_analysis_integration()
        
        print("\n" + "="*60)
        # # Removed print spam: "...
        print("\nKey Features Demonstrated:")
        # Removed print spam: "  ...
        print("  📋 Comprehensive multi-section reports")  
        print("  💾 Multiple export formats (JSON, CSV, Markdown, Text)")
        # Removed print spam: "  ...")
        # Removed print spam: "  ...
        print("  🧠 Integration with analysis engine for deep insights")
        # Removed print spam: "  ...
        
        # Removed print spam: "\n...
        print("   symbolic AI reasoning and human understanding through multiple")
        print("   presentation formats and visualization styles!")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()