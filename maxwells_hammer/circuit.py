from typing import List
from dataclasses import dataclass, field
from .components import Component, VoltageSource, Inductor

@dataclass
class Circuit:
    components: List[Component] = field(default_factory=list)
    
    def add(self, component: Component):
        self.components.append(component)
        
    @property
    def num_nodes(self) -> int:
        """Returns the number of nodes (including ground 0)."""
        max_node = 0
        for c in self.components:
            max_node = max(max_node, c.node1, c.node2)
        return max_node + 1

    @property
    def num_v_sources(self) -> int:
        return sum(1 for c in self.components if isinstance(c, (VoltageSource, Inductor)))
