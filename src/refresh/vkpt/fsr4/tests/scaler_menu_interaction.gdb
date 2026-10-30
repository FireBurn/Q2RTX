# Live integration test. Run a Debug build under gdb --batch -x this-file,
# with an isolated XDG_DATA_HOME and matching Q2RTX_DATA_DIR assets.
# Required engine arguments:
# +set vk_validation 1 +set vid_fullscreen 0 +set vid_geometry 960x540
# +set flt_upscaler 2 +set flt_fsr4_dynamic_resolution 0
# +set flt_fsr4_sharpening 0 +set flt_fsr3_sharpening 0
# +set ui_scaler_fsr315_sharpness 0 +set flt_frame_generation 0
# +pushmenu temporal_settings
# Captures catalog_interaction.png and exits. This drives the engine's real
# keyboard handlers; it does not simulate physical controller/OS input.
set pagination off
set confirm off
break UI_Draw if uis.activeMenu != 0
commands
silent
disable 1
set $m = uis.activeMenu
python assert gdb.parse_and_eval('$m->name').string() == 'temporal_settings'
python assert int(gdb.parse_and_eval('((menuSpinControl_t*)$m->items[0])->cvar->integer')) == 2
call Menu_SetFocus((menuCommon_t*)$m->items[4])
call UI_KeyEvent(131, 1)
call UI_KeyEvent(131, 1)
call Menu_SetFocus((menuCommon_t*)$m->items[6])
call UI_KeyEvent(131, 1)
python assert abs(float(gdb.parse_and_eval('((menuSlider_t*)$m->items[4])->cvar->value')) - 0.1) < 0.0001
python assert int(gdb.parse_and_eval('((menuSpinControl_t*)$m->items[6])->cvar->integer')) == 1
python assert int(gdb.parse_and_eval('((menuCommon_t*)$m->items[3])->flags')) & 16
call Menu_SetFocus((menuCommon_t*)$m->items[0])
call UI_KeyEvent(131, 1)
python assert int(gdb.parse_and_eval('((menuSpinControl_t*)$m->items[0])->cvar->integer')) == 3
python assert int(gdb.parse_and_eval('((menuCommon_t*)$m->items[4])->flags')) & 16
python assert not int(gdb.parse_and_eval('((menuCommon_t*)$m->items[7])->flags')) & 16
python assert float(gdb.parse_and_eval('((menuSlider_t*)$m->items[8])->cvar->value')) == 0
call Menu_SetFocus((menuCommon_t*)$m->items[8])
call UI_KeyEvent(131, 1)
call Menu_SetFocus((menuCommon_t*)$m->items[0])
call UI_KeyEvent(130, 1)
python assert abs(float(gdb.parse_and_eval('((menuSlider_t*)$m->items[4])->cvar->value')) - 0.1) < 0.0001
python assert int(gdb.parse_and_eval('((menuCommon_t*)$m->items[3])->flags')) & 16
call UI_PopMenu()
call UI_PushMenu($m)
python assert abs(float(gdb.parse_and_eval('((menuSlider_t*)$m->items[4])->cvar->value')) - 0.1) < 0.0001
python assert abs(float(gdb.parse_and_eval('((menuSlider_t*)$m->items[8])->cvar->value')) - 0.05) < 0.0001
printf "MENU_TEST passed: immediate switching, contextual visibility, independent preferences, reopen persistence\n"
call Cbuf_AddText(&cmd_buffer, "wait 30; screenshotpng catalog_interaction; wait 5; quit\n")
continue
end
run
