import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from dataclasses import dataclass
from typing import Callable, Any

@register_pytree_node_class
@dataclass
class Component:
    """Base class for circuit components."""
    node1: int
    node2: int
    name: str

    def tree_flatten(self):
        # Default flattening for dataclasses with static name/nodes if they are structural
        # However, for now, let's treat nodes as part of the structure (aux) and values as leaves
        # This might vary per component.
        return (), (self.node1, self.node2, self.name)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*aux_data)

@register_pytree_node_class
@dataclass
class Resistor(Component):
    resistance: float  # Value in Ohms

    def tree_flatten(self):
        return (self.resistance,), (self.node1, self.node2, self.name)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children[0], *aux_data)

@register_pytree_node_class
@dataclass
class Capacitor(Component):
    capacitance: float # Value in Farads

    def tree_flatten(self):
        return (self.capacitance,), (self.node1, self.node2, self.name)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children[0], *aux_data)

@register_pytree_node_class
@dataclass
class Inductor(Component):
    inductance: float # Value in Henrys

    def tree_flatten(self):
        return (self.inductance,), (self.node1, self.node2, self.name)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children[0], *aux_data)

@register_pytree_node_class
@dataclass
class VoltageSource(Component):
    # For simple DC or time-dependent. 
    # If dc_value is a float, it's DC. 
    # To support time-varying, we might need a more complex structure, 
    # but for JAX scans, passing a function might be tricky unless it's pure.
    # We'll stick to a concrete value (which can be updated in a scan) or a simple DC value for now. 
    # Let's assume constant DC for basic solve, and for transient we might handle it differently.
    dc_value: float 

    def tree_flatten(self):
        return (self.dc_value,), (self.node1, self.node2, self.name)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children[0], *aux_data)

@register_pytree_node_class
@dataclass
class CurrentSource(Component):
    dc_value: float 

    def tree_flatten(self):
        return (self.dc_value,), (self.node1, self.node2, self.name)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children[0], *aux_data)
