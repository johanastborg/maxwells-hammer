import jax
import jax.numpy as jnp
from .components import Resistor, Capacitor, Inductor, VoltageSource, CurrentSource
from .circuit import Circuit

def assemble_mna_matrices(circuit: Circuit):
    """
    Assembles the G, C matrices and RHS vector for modified nodal analysis.
    Equation: G * x + C * dx/dt = u
    
    x size: num_nodes - 1 (voltages) + num_v_sources (currents)
    Note: Node 0 is reference (ground) and is excluded from the matrix rows/cols usually,
    or handled by constraints. Here we eliminate node 0.
    """
    
    num_nodes = circuit.num_nodes - 1 # Exclude ground
    
    # Identify auxiliary variables (currents through V-sources and Inductors)
    # We assign an index to each V-source/Inductor
    v_source_indices = {}
    idx_counter = num_nodes
    for c in circuit.components:
        if isinstance(c, (VoltageSource, Inductor)):
            v_source_indices[id(c)] = idx_counter
            idx_counter += 1
            
    total_vars = idx_counter
    
    # Initialize matrices
    G = jnp.zeros((total_vars, total_vars))
    C = jnp.zeros((total_vars, total_vars))
    u_dc = jnp.zeros(total_vars) # Constant sources
    
    # Helper to add to matrix safely
    def add_val(mat, r, c, val):
        if 0 <= r < total_vars and 0 <= c < total_vars:
            mat = mat.at[r, c].add(val)
        return mat
        
    def add_rhs(vec, r, val):
        if 0 <= r < total_vars:
            vec = vec.at[r].add(val)
        return vec

    # Note: Node indices in components are 0-based. 
    # Matrix indices 0..(num_nodes-1) correspond to Nodes 1..num_nodes.
    # Node 0 is Ground, so current invalid.
    # We map component node to matrix index: node i -> i-1. If node==0, ignore (ground).

    for comp in circuit.components:
        n1 = comp.node1
        n2 = comp.node2
        
        # Resistor: adds to G
        if isinstance(comp, Resistor):
            g = 1.0 / comp.resistance
            # Nodes
            r1, r2 = n1 - 1, n2 - 1
            
            # Diagonal
            G = add_val(G, r1, r1, g)
            G = add_val(G, r2, r2, g)
            # Off-diagonal
            G = add_val(G, r1, r2, -g)
            G = add_val(G, r2, r1, -g)

        # Capacitor: adds to C
        elif isinstance(comp, Capacitor):
            c = comp.capacitance
            r1, r2 = n1 - 1, n2 - 1
            
            C = add_val(C, r1, r1, c)
            C = add_val(C, r2, r2, c)
            C = add_val(C, r1, r2, -c)
            C = add_val(C, r2, r1, -c)
            
        # Voltage Source: adds new row/col, adds to G (1s) and RHS
        elif isinstance(comp, VoltageSource):
            idx = v_source_indices[id(comp)]
            r1, r2 = n1 - 1, n2 - 1
            
            # MNA stamp: 
            # Row 'idx' (equation for V source): V(n1) - V(n2) = V_src
            G = add_val(G, idx, r1, 1.0)
            G = add_val(G, idx, r2, -1.0)
            
            # Col 'idx' (current variable entering KCL): +I at n1, -I at n2
            G = add_val(G, r1, idx, 1.0)
            G = add_val(G, r2, idx, -1.0)
            
            u_dc = add_rhs(u_dc, idx, comp.dc_value)

        # Inductor: adds new row/col, adds to C (L at diagonal), G (1s)
        elif isinstance(comp, Inductor):
            idx = v_source_indices[id(comp)]
            r1, r2 = n1 - 1, n2 - 1
            
            # V(n1) - V(n2) - L * dI/dt = 0  => V(n1) - V(n2) = L * dI/dt
            # MNA usually puts the L term in C matrix at (idx, idx) with negative sign if moving to RHS?
            # Standard form: G*x + C*x' = u
            # Row idx: V(n1) - V(n2) - L * x'[idx] = 0
            # So: G terms for V(n1), V(n2). C term for L at (idx, idx) is -L.
            
            G = add_val(G, idx, r1, 1.0)
            G = add_val(G, idx, r2, -1.0)
            
            G = add_val(G, r1, idx, 1.0)
            G = add_val(G, r2, idx, -1.0)
            
            C = add_val(C, idx, idx, -comp.inductance)
            
        elif isinstance(comp, CurrentSource):
            r1, r2 = n1 - 1, n2 - 1
            # Current leaves n1, enters n2 ? Convention check.
            # Usually independent source pointing n1->n2 means current flows out of n1 into n2?
            # If so, KCL at n1: ... + I_src = 0 => ... = -I_src
            # Let's assume defined positive current flows n1 -> n2.
            # KCL n1 (sum currents leaving = 0): I_src leaves node 1.
            # So term is +I_src on LHS, or -I_src on RHS.
            
            u_dc = add_rhs(u_dc, r1, -comp.dc_value)
            u_dc = add_rhs(u_dc, r2, comp.dc_value)

    return G, C, u_dc

@jax.jit
def solve_dc(G, u):
    """Solves G * x = u for DC operating point (assuming C * dx/dt = 0)."""
    return jax.numpy.linalg.solve(G, u)

def solve_transient(circuit: Circuit, t_span, dt, x0=None):
    """
    Solves transient analysis using Backward Euler.
    (C/dt + G) * x_next = C/dt * x_prev + u(t_next)
    """
    
    # We need to assemble matrices inside the function if we want to differentiate 
    # with respect to component parameters.
    # However, 'circuit' is a Python object. 
    # For JIT, we should separate 'data' from 'structure'.
    # For this simple example, we'll re-assemble every time or rely on JAX to trace.
    
    G, C, u_dc = assemble_mna_matrices(circuit)
    
    num_steps = int((t_span[1] - t_span[0]) / dt)
    timestamps = jnp.linspace(t_span[0], t_span[1], num_steps)
    
    total_vars = G.shape[0]
    if x0 is None:
        x0 = jnp.zeros(total_vars)
        
    # Pre-compute LHS for Backward Euler
    # (C/dt + G) * x_next = C/dt * x_prev + u
    A_eff = (C / dt) + G
    
    # Factorize A_eff once if G/C are constant
    solve_op = jax.scipy.linalg.lu_factor(A_eff)
    
    def step_fn(x_prev, t):
        # RHS = (C/dt)*x_prev + u(t)
        # Assuming dc sources for now
        u_t = u_dc 
        rhs = (C @ x_prev) / dt + u_t
        
        x_next = jax.scipy.linalg.lu_solve(solve_op, rhs)
        return x_next, x_next
    
    final_x, trajectory = jax.lax.scan(step_fn, x0, timestamps)
    
    return timestamps, trajectory
