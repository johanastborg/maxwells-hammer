import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from maxwells_hammer.components import Resistor, Capacitor, VoltageSource
from maxwells_hammer.circuit import Circuit
from maxwells_hammer.solver import solve_transient, assemble_mna_matrices

def main():
    # 1. Define Circuit
    # Simple RC Circuit: V1 --- R1 --- n1 --- C1 --- gnd (0)
    # V1 connected between n2 (input) and gnd? Or just V1 node1=1, node2=0.
    
    # Let's map: 
    # Node 0: GND
    # Node 1: V_source positive terminal (connected to R)
    # Node 2: Connection between R and C
    
    # Circuit:
    # V_src connected between 1 and 0.
    # R connected between 1 and 2.
    # C connected between 2 and 0.
    
    c = Circuit()
    c.add(VoltageSource(name="V1", node1=1, node2=0, dc_value=10.0)) # 10V Step
    c.add(Resistor(name="R1", node1=1, node2=2, resistance=1000.0)) # 1k Ohm
    c.add(Capacitor(name="C1", node1=2, node2=0, capacitance=1e-6)) # 1 uF
    
    print(f"Num nodes (including GND): {c.num_nodes}")
    
    # 2. Transient Simulation
    # RC time constant = R*C = 1k * 1u = 1ms.
    # Simulate for 5ms (5 tau).
    t_end = 0.005
    dt = 1e-5 # 10 us steps
    
    print("Starting simulation...")
    timestamps, trajectory = solve_transient(c, (0.0, t_end), dt)
    print("Simulation complete.")
    
    # Trajectory shape: (num_steps, num_vars)
    # Vars indices: 0->Node1 (V_src), 1->Node2 (V_cap), 2->Current through V1 (aux)
    
    # Note: assemble_mna_matrices assembles indices.
    # G matrix size: num_nodes-1 + num_v_sources.
    # Here num_nodes=3 (0,1,2). Vars: Node1, Node2, I_V1.
    # Variable mapping relies on order. 
    # Nodes are 1..N mapped to 0..N-1.
    # Node 1 -> index 0.
    # Node 2 -> index 1.
    # V_source current -> index 2.
    
    v_in = trajectory[:, 0]
    v_out = trajectory[:, 1]
    
    # 3. Plot
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps * 1000, v_in, label='V_in (Node 1)')
        plt.plot(timestamps * 1000, v_out, label='V_out (Node 2, Capacitor)')
        plt.title('RC Circuit Step Response (JAX Solver)')
        plt.xlabel('Time (ms)')
        plt.ylabel('Voltage (V)')
        plt.grid(True)
        plt.legend()
        plt.savefig('rc_response.png')
        print("Plot saved to rc_response.png")
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    main()
