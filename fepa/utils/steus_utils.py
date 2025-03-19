import os

def make_test_mdp():
    mdp = """
    integrator              = md
    dt                      = 0.002
    nsteps                  = 250000  ; 500 ps
    nstxout-compressed      = 5000
    nstxout                 = 0
    nstvout                 = 0
    nstfout                 = 0
    nstcalcenergy           = 50
    nstenergy               = 50
    nstlog                  = 5000
    ;
    cutoff-scheme           = Verlet
    nstlist                 = 20
    rlist                   = 0.9
    vdwtype                 = Cut-off
    vdw-modifier            = None
    DispCorr                = EnerPres
    rvdw                    = 0.9
    coulombtype             = PME
    rcoulomb                = 0.9
    ;
    tcoupl                  = v-rescale
    tc_grps                 = POPC_FIP SOLV
    tau_t                   = 1.0 1.0
    ref_t                   = {T} {T}
    ;
    pcoupl                  = C-rescale
    pcoupltype              = semiisotropic 
    tau_p                   = 5.0
    compressibility         = 4.5e-5  4.5e-5
    ref_p                   = 1.0     1.0
    refcoord_scaling        = com
    ;
    constraints             = h-bonds
    constraint_algorithm    = LINCS
    continuation            = no
    gen-vel                 = yes
    gen-temp                = {T}
    ;
    nstcomm                 = 100
    comm_mode               = linear
    comm_grps               = POPC_FIP SOLV

    ; Pull code
    pull                     = yes
    pull_ncoords             = 1
    pull_ngroups             = 2
    pull_group1_name         = FIP
    pull_group2_name         = POPC
    pull_group2_pbcatom      = 995
    pull_pbc_ref_prev_step_com = yes
    pull_coord1_type         = umbrella
    pull_coord1_geometry     = direction
    pull_coord1_vec          = 0 0 1
    pull_coord1_dim          = N N Y  ; pull along z
    pull_coord1_groups       = 1 2
    pull_coord1_start        = yes
    pull_coord1_rate         = 0.0   ; restrain in place
    pull_coord1_k            = 1000  ; kJ mol^-1 nm^-2
    pull_nstfout             = 50

    """

    # These sometimes very small steps are tested because I might want to run STeUS
    # either with delta T of 4 or of 5, and I'll need these simulations to calculate
    # the weights, so I might as well run them all now while testing stability.
    temps = ["310", "316", "322", "328", "334", "340", "346", "352", "358"]

    for temp in temps:
        with open(f"T{temp}.mdp", "w") as f:
            f.write(mdp.format(T=temp))
        os.system(f"gmx grompp -f T{temp}.mdp -c ../../window_prep/alchembed13.gro -r ../../window_prep/alchembed13.gro -n ../../window_prep/index.ndx -p ../../window_prep/topol.top -o runs/T{temp}")