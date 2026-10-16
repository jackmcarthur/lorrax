"""Time-dependent adiabatic GW (TD-aGW) on the ISDF basis.

``tdagw.model``          the interaction model ΔΣ_k[δρ] (``build_td_agw_apply``)
                         and its input bundle (``load_td_model``,
                         ``assemble_td_model``);
``tdagw.kernel_export``  the dense kernel K_{kk',nm,n'm'} as a TDGW ``kernel.h5``;
``tdagw.tdgw_bridge``    TDGW's propagator with its self-energy update replaced
                         by the model.

Theory: ``docs/theory/td_agw.md``.
"""
