binary_count_keys = ['TP', 'FP', 'TN', 'FN']

# labels (prefix "l_"), mainly used for ternary classification (but in
# the early days I used it for binary also)
lb_green = 1
lb_amber = 2
lb_red = 3
lb_other = -1

# phases (prefix "p_"), mainly used for binary classification
# "ph_unknown" e.g. at start of data collection, where we don't know
# what happened before
ph_healthy = 11
ph_prodromal = 12
ph_onset = 13
ph_exacerbating = 14
ph_recovery = 15
ph_unknown = 16

# alerts (prefix "a_"), mainly used for binary classification
al_pos = 1
al_neg = 0
al_unk  = -1

# codes for FSM states, as of writing not sure if this is used (but
# the motivation is that you can go from transition back to normal,
# meaning that FSM states are not directly interpretable as a phase,
# especially in the context of TEW curves)
# used for both binary/ternary classification
fsm_normal = 21
fsm_transition = 22
fsm_exacerbating = 23
