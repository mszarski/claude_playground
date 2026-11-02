N=/tmp/claude-0/-home-user-claude-playground/9fa36f9b-b558-5a88-94f2-47889dc83674/scratchpad/sans
cd /home/user/claude_playground/ductus
run() { g=$1; shift; out=$(python $N/cd2.py $g "$@" 2>&1 | tail -1); echo "$g: $out"; d=$(echo "$out" | sed -n 's/^end [0-9.]* //p'); if [ -n "$d" ] && [ "$d" != "{}" ]; then python $N/apply.py "$d"; fi; }
run eight eight_overlap:0.05 eight_waist:8 eight_top_w:10 eight_w:8 eight_thick_k:0.03 eight_sp_w:0.05
run seven seven_stem:4 seven_top_dir:5 seven_foot_dir:5 seven_foot_l:8 seven_foot_r:8 seven_w:8 seven_bar_t:4
run g gs_dfar:6 gs_bot:8 gs_w:8 gs_end_y:8 gs_end_dir:6 gs_end_w:0.1 descender_g:6
run sterling, pound_foot_x:0.02 pound_foot_dir:5 pound_rise_x:0.02 pound_rise_y:0.03 pound_top_x:0.03 pound_end_x:0.03 pound_end_y:0.02 pound_end_w:0.1 pound_bar_l:0.02 pound_bar_r:0.03 pound_base_k:0.1 pound_bar_y:8 pound_w:10
run asciitilde, tl_len:8 tl_amp:5 tl_w:4 tl_dir:5 atilde_dy:6
run asciicircum, caret_w:8 caret_h:8 caret_y0:8 caret_t:4
run cedilla,ccedilla,Ccedilla ced_len:8 ced_drop:8 ced_neck:6 ced_thin:3 acc_dx_c:5 acc_dx_C:5
run question, quest_width:8 quest_w:4 quest_thin:3 quest_end_y:8 quest_neck_x:0.02 quest_neck_y:8 quest_bot:5 quest_bot_w:0.05 quest_tension:0.05 quest_top_x:0.02 quest_right_y:8 quest_neck_dir:4 quest_bot_dx:4
run at, at_w:10 at_h:10 at_bot:6 at_ring_t:3 at_in_w:8 at_in_h:8 at_in_x:0.02 at_in_y:8 at_root_dir:6 at_end_x:0.03 at_end_y:0.02 at_end_dir:6 at_right_x:0.02 at_pen_k:0.05
run braceleft,braceright brace_w:8 brace_pen:4 brace_x:0.02 brace_end_w:0.1 brace_beak_w:0.1 brace_beak_x:4
run ampersand, amp_w:8 amp_arm_x:0.02 amp_arm_y:8 amp_loop_x:0.02 amp_loop_w:8 amp_waist_y:8 amp_leg_x:0.02 amp_pen:3 amp_loop_y:8 amp_loop_lx:6 amp_left_y:8 amp_bot_x:0.02 amp_waist_dir:4 amp_leg_dir:3
python $N/tidy.py
python $N/exp.py examples/sans/font.py w3 2>&1 | tail -1
