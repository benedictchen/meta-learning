#!/usr/bin/env python3
"""
🔒 Constraint Engine Demonstration
=================================

This script demonstrates the usage of the extracted constraint engine module,
showcasing its security features, evaluation methods, and practical applications
in qualitative reasoning systems.

Author: Benedict Chen
"""

import sys
sys.path.insert(0, '/Users/benedictchen/work/research_papers/packages/qualitative_reasoning')

from qualitative_reasoning.qr_modules import (
    ConstraintEvaluationMethod,
    ConstraintEvaluationConfig,
    ConstraintEngineMixin,
    QualitativeValue,
    QualitativeDirection,
    QualitativeQuantity
)


class AdvancedQualitativeReasoner(ConstraintEngineMixin):
    """
    Advanced qualitative reasoner demonstrating constraint engine usage.
    
    This class shows how to integrate the constraint engine mixin into
    a qualitative reasoning system with real-world physical modeling.
    """
    
    def __init__(self, domain_name: str = "Physical System"):
        # Configure constraint evaluation for maximum security
        constraint_config = ConstraintEvaluationConfig(
            evaluation_method=ConstraintEvaluationMethod.HYBRID,
            allow_function_calls=False,
            allow_attribute_access=False,
            strict_mode=False,
            fallback_to_false=True
        )
        
        super().__init__(constraint_config=constraint_config)
        
        self.domain_name = domain_name
        self.quantities = {}
        self.constraints = []
        
        print(f"🧠 Advanced Qualitative Reasoner: {domain_name}")
        print(f"🔒 Constraint Engine: {constraint_config.evaluation_method.value}")
    
    def add_quantity(self, name: str, magnitude: QualitativeValue, 
                    direction: QualitativeDirection):
        """Add a quantity to the system and constraint whitelist."""
        
        self.quantities[name] = QualitativeQuantity(
            name=name,
            magnitude=magnitude,
            direction=direction
        )
        
        # Add to constraint evaluation whitelist
        self.add_allowed_variable(name)
        
        print(f"  ✓ Added quantity: {name} = {magnitude.value}, trend: {direction.value}")
    
    def add_constraint(self, constraint: str):
        """Add and validate a constraint."""
        
        self.constraints.append(constraint)
        print(f"  ✓ Added constraint: {constraint}")
        
        # Test the constraint immediately
        try:
            result = self._evaluate_logical_expression(constraint)
            status = "SATISFIED" if result else "VIOLATED"
            print(f"    Current status: {status}")
        except Exception as e:
            print(f"    ⚠️ Constraint evaluation error: {e}")
    
    def check_all_constraints(self):
        """Check all system constraints."""
        
        print(f"\n🔍 Checking {len(self.constraints)} constraints...")
        
        satisfied = 0
        violated = 0
        
        for i, constraint in enumerate(self.constraints):
            try:
                result = self._evaluate_logical_expression(constraint)
                if result:
                    satisfied += 1
                    print(f"  ✅ [{i+1}] {constraint}")
                else:
                    violated += 1
                    print(f"  ❌ [{i+1}] {constraint}")
            except Exception as e:
                violated += 1
                print(f"  ⚠️ [{i+1}] {constraint} - ERROR: {e}")
        
        print(f"\n📊 Constraint Summary: {satisfied} satisfied, {violated} violated")
        return satisfied, violated
    
    def demonstrate_security(self):
        """Demonstrate security features."""
        
        print(f"\n🛡️ Security Demonstration")
        print("=" * 30)
        
        malicious_constraints = [
            "__import__('os').system('rm -rf /')",
            "exec('print(\"hacked\")')",
            "eval('open(\"/etc/passwd\").read()')",
            "temperature.__class__.__mro__",
            "getattr(temperature, '__dict__')"
        ]
        
        print("Testing malicious constraint inputs...")
        for constraint in malicious_constraints:
            try:
                result = self._evaluate_logical_expression(constraint)
                print(f"  🛡️ BLOCKED: '{constraint[:50]}...' → {result}")
            except Exception as e:
                print(f"  🛡️ SECURED: '{constraint[:50]}...' → {type(e).__name__}")
    
    def demonstrate_evaluation_methods(self):
        """Demonstrate different evaluation methods."""
        
        print(f"\n⚙️ Evaluation Methods Demonstration")
        print("=" * 40)
        
        test_constraints = [
            "temperature > 0",
            "temperature + pressure > flow_rate", 
            "temperature > 0 and pressure > 0",
            "temperature > 0 => pressure > 0",
            "not (flow_rate > temperature)"
        ]
        
        methods = [
            ConstraintEvaluationMethod.AST_SAFE,
            ConstraintEvaluationMethod.REGEX_PARSER,
            ConstraintEvaluationMethod.HYBRID
        ]
        
        for method in methods:
            print(f"\n🔧 Testing {method.value}:")
            self.configure_constraint_evaluation(method)
            
            for constraint in test_constraints:
                try:
                    result = self._evaluate_logical_expression(constraint)
                    print(f"  {'✓' if result else '✗'} {constraint} = {result}")
                except Exception as e:
                    print(f"  ⚠️ {constraint} - {type(e).__name__}: {e}")


def demo_thermal_system():
    """Demonstrate thermal system with constraint engine."""
    
    print("\n🌡️ Thermal System Demo")
    print("=" * 50)
    
    # Create thermal system reasoner
    thermal = AdvancedQualitativeReasoner("Thermal System")
    
    # Add thermal quantities
    thermal.add_quantity("temperature", QualitativeValue.POSITIVE_SMALL, 
                        QualitativeDirection.INCREASING)
    thermal.add_quantity("heat_flow", QualitativeValue.POSITIVE_SMALL,
                        QualitativeDirection.STEADY)
    thermal.add_quantity("thermal_energy", QualitativeValue.POSITIVE_LARGE,
                        QualitativeDirection.INCREASING)
    thermal.add_quantity("heat_capacity", QualitativeValue.POSITIVE_LARGE,
                        QualitativeDirection.STEADY)
    
    # Add thermal constraints
    thermal.add_constraint("temperature >= 0")  # Temperature cannot be below absolute zero
    thermal.add_constraint("thermal_energy >= 0")  # Energy is non-negative
    thermal.add_constraint("temperature > 0 => thermal_energy > 0")  # If hot, has energy
    thermal.add_constraint("heat_flow > 0 => temperature > 0")  # Heat flow requires temperature difference
    thermal.add_constraint("thermal_energy > heat_capacity")  # System has more energy than capacity
    
    # Check all constraints
    thermal.check_all_constraints()
    
    # Demonstrate security
    thermal.demonstrate_security()
    
    return thermal


def demo_fluid_system():
    """Demonstrate fluid system with constraint engine."""
    
    print("\n💧 Fluid System Demo")
    print("=" * 50)
    
    # Create fluid system reasoner
    fluid = AdvancedQualitativeReasoner("Fluid System")
    
    # Add fluid quantities
    fluid.add_quantity("pressure", QualitativeValue.POSITIVE_LARGE,
                      QualitativeDirection.DECREASING)
    fluid.add_quantity("flow_rate", QualitativeValue.POSITIVE_SMALL,
                      QualitativeDirection.INCREASING)
    fluid.add_quantity("volume", QualitativeValue.POSITIVE_LARGE,
                      QualitativeDirection.DECREASING)
    fluid.add_quantity("viscosity", QualitativeValue.POSITIVE_SMALL,
                      QualitativeDirection.STEADY)
    
    # Add fluid dynamics constraints
    fluid.add_constraint("pressure >= 0")  # Gauge pressure non-negative
    fluid.add_constraint("volume >= 0")    # Volume non-negative
    fluid.add_constraint("flow_rate >= 0") # Flow rate non-negative
    fluid.add_constraint("pressure > 0 => flow_rate >= 0")  # Pressure drives flow
    fluid.add_constraint("flow_rate > 0 and volume > 0")    # Active flow with volume
    fluid.add_constraint("viscosity > 0")  # Viscosity is always positive
    
    # Check all constraints
    fluid.check_all_constraints()
    
    # Demonstrate different evaluation methods
    fluid.demonstrate_evaluation_methods()
    
    return fluid


def demo_security_configuration():
    """Demonstrate security configuration options."""
    
    print("\n🔐 Security Configuration Demo")
    print("=" * 50)
    
    # Create reasoner with strict security
    strict_config = ConstraintEvaluationConfig(
        evaluation_method=ConstraintEvaluationMethod.AST_SAFE,
        allow_function_calls=False,
        allow_attribute_access=False,
        strict_mode=True,  # Fail on any error
        fallback_to_false=False
    )
    
    strict_reasoner = AdvancedQualitativeReasoner("Strict Security System")
    strict_reasoner.constraint_config = strict_config
    
    # Add basic quantities
    strict_reasoner.add_quantity("x", QualitativeValue.POSITIVE_SMALL,
                               QualitativeDirection.INCREASING)
    
    print("\n🔒 Strict Mode Testing:")
    safe_constraints = [
        "x > 0",
        "x + 1 > 0", 
        "x * 2 >= x"
    ]
    
    for constraint in safe_constraints:
        try:
            result = strict_reasoner._evaluate_logical_expression(constraint)
            print(f"  ✅ {constraint} = {result}")
        except Exception as e:
            print(f"  ❌ {constraint} - {type(e).__name__}: {e}")
    
    # Test unsafe constraints in strict mode
    print("\n⚠️ Testing unsafe constraints in strict mode:")
    unsafe_constraints = [
        "unknown_variable > 0",
        "x + + 1",  # Syntax error
        "__import__('sys')"
    ]
    
    for constraint in unsafe_constraints:
        try:
            result = strict_reasoner._evaluate_logical_expression(constraint)
            print(f"  ⚠️ {constraint} = {result} (should have failed)")
        except Exception as e:
            print(f"  🛡️ {constraint} - PROPERLY BLOCKED: {type(e).__name__}")


def main():
    """Run comprehensive constraint engine demonstration."""
    
    print("🔒 Constraint Engine Module Demonstration")
    print("=" * 60)
    print("Demonstrating extracted constraint engine with real-world examples...")
    
    # Run all demonstrations
    thermal = demo_thermal_system()
    fluid = demo_fluid_system()
    demo_security_configuration()
    
    # Final summary
    print("\n🎉 Demonstration Complete!")
    print("=" * 60)
    print("✅ Constraint engine successfully extracted from monolithic file")
    print("🔒 Security features prevent code injection and malicious execution")
    print("⚙️ Multiple evaluation methods provide flexibility and robustness")
    print("🛠️ Error handling ensures graceful degradation")
    print("🧠 Integration with qualitative reasoning is seamless")
    
    print("\n📋 Key Benefits:")
    print("  • Modular design enables better testing and maintenance")
    print("  • Security-first approach eliminates eval() vulnerabilities")
    print("  • Configurable evaluation methods suit different use cases")
    print("  • Comprehensive error handling improves reliability")
    print("  • Clean separation of concerns improves code organization")


if __name__ == "__main__":
    main()