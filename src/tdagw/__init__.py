"""Time-dependent adiabatic GW (TD-aGW) on the ISDF basis.

``tdagw.model``          the interaction model ΔΣ_k[δρ] (``build_td_agw_apply``)
                         and its input bundle (``load_td_model``,
                         ``assemble_td_model``);
``tdagw.kernel_export``  the dense kernel K_{kk',nm,n'm'} as a TDGW ``kernel.h5``;
``tdagw.tdgw_bridge``    TDGW's propagator with its self-energy update replaced
                         by the model.

The JAX propagator (driver ``python -m tdagw.tdgw_jax -i tdgw.toml``):

``tdagw.eom``            TDGW's equation of motion on the sharded k axis;
``tdagw.self_energy``    the ΔΣ[ρ] contract and its models (none, TDGW
                         ``kernel.h5``, low rank, the ISDF model above);
``tdagw.propagate``      the integrators (``solvers.time_integrators``) and the
                         jitted chunk program;
``tdagw.tdgw_io``        TDGW header, deck drive and ``tdgw.h5`` output;
``tdagw.td_config``      the deck schema and its refusals;
``tdagw.td_system``      the TD system built from a GW restart bundle
                         (``[lorrax] system = "bundle"``) and its receipts;
``tdagw.stages``         the driver's stages (system, model, outputs, run);
``tdagw.tdgw_jax``       the driver.

Theory: ``docs/theory/td_agw.md`` (the interaction) and
``docs/theory/td_agw_propagation.md`` (the propagation).
"""
