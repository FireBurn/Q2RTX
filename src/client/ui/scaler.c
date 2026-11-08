/* SPDX-License-Identifier: GPL-2.0-or-later
 * Q2RTX bindings and widgets for the reusable scaler catalog.
 */
#include "ui.h"

#if REF_VKPT
#include "ffx_vk_scaler_catalog.h"

static cvar_t *model_cvar;
static cvar_t *available_models;
static cvar_t *settings[FFX_VK_SCALER_COUNT][FFX_VK_SCALER_OPTION_COUNT];

static void update_description(menuFrameWork_t *menu)
{
    menuCommon_t *selector = menu->items[0];
    const FfxVkScalerModel *model = ffxVkScalerModel(model_cvar->integer);
    Z_Free(selector->status);
    const char *description = model ? model->description : "Unknown model selection.";
    if (model) {
        cvar_t *reason = Cvar_FindVar(va("flt_scaler_%s_reason", model->key));
        if (reason && reason->string[0])
            description = reason->string;
    }
    selector->status = UI_CopyString(description);
    if (selector->flags & QMF_HASFOCUS)
        menu->status = selector->status;
}

/* Only engine bindings belong here. Labels, values and applicability come
 * from the library, which has no knowledge of Q2RTX cvars or menus.
 */
static const char *binding(int model, int option)
{
    switch (option) {
    case FFX_VK_SCALER_OPTION_QUALITY: return "flt_fsr_quality";
    case FFX_VK_SCALER_OPTION_SHARPNESS:
        return model == FFX_VK_SCALER_FSR4_NEURAL ?
            "flt_fsr4_sharpening" : "flt_fsr3_sharpening";
    case FFX_VK_SCALER_OPTION_AUTO_EXPOSURE: return "flt_fsr4_auto_exposure";
    case FFX_VK_SCALER_OPTION_DYNAMIC_RESOLUTION: return "flt_fsr4_dynamic_resolution";
    case FFX_VK_SCALER_OPTION_VARIANT: return "flt_fsr4_variant";
    default: return NULL;
    }
}

static int variant_string_to_index(const char *s)
{
    if (!s || !s[0])
        return 0;
    if (!Q_stricmp(s, "412_i8_2") || !Q_stricmp(s, "i8_2") ||
        !Q_stricmp(s, "navi2") || !Q_stricmp(s, "rdna2") || !strcmp(s, "1"))
        return 1;
    if (!Q_stricmp(s, "412_lite_i8") || !Q_stricmp(s, "lite_i8") ||
        !Q_stricmp(s, "lite") || !strcmp(s, "2"))
        return 2;
    if (!Q_stricmp(s, "412_i8") || !Q_stricmp(s, "rdna3") || !strcmp(s, "3"))
        return 3;
    if (!Q_stricmp(s, "v07_i8") || !Q_stricmp(s, "v07") || !strcmp(s, "4"))
        return 4;
    return 0;
}

static const char *variant_index_to_string(int index)
{
    static const char *variants[] = { "auto", "412_i8_2", "412_lite_i8", "412_i8", "v07_i8" };
    if (index >= 0 && index < 5)
        return variants[index];
    return "auto";
}

static bool option_visible(menuCommon_t *item)
{
    int model = item->id / FFX_VK_SCALER_OPTION_COUNT;
    int option = item->id % FFX_VK_SCALER_OPTION_COUNT;
    float values[FFX_VK_SCALER_OPTION_COUNT] = { 0 };
    for (int i = 0; i < FFX_VK_SCALER_OPTION_COUNT; ++i)
        if (settings[model][i])
            values[i] = settings[model][i]->value;
    return model_cvar->integer == model &&
        ffxVkScalerOptionConfigurable(model, option, values,
            (unsigned)available_models->integer);
}

static bool controller_visible(menuCommon_t *item)
{
    (void)item;
    int model = model_cvar->integer;
    int option = FFX_VK_SCALER_OPTION_DYNAMIC_RESOLUTION;
    return ffxVkScalerOptionConfigurable(model, option, NULL,
        (unsigned)available_models->integer) && settings[model][option] &&
        settings[model][option]->integer != 0;
}

static menuSound_t open_controller(menuCommon_t *item)
{
    (void)item;
    Cbuf_AddText(&cmd_buffer, "pushmenu dynamic_resolution\n");
    return QMS_SILENT;
}

static void apply_settings(void)
{
    int model = model_cvar->integer;
    if (!ffxVkScalerModel(model))
        return;
    /* Explicit Off must not reactivate the deprecated FSR enable alias. */
    Cvar_Set("flt_fsr_enable", "0");
    for (int i = 0; i < FFX_VK_SCALER_OPTION_COUNT; ++i) {
        cvar_t *value = settings[model][i];
        if (!value)
            continue;
        if (!ffxVkScalerOptionValueValid(model, i, value->value))
            Cvar_SetValue(value, ffxVkScalerOption(i)->defaultValue, FROM_MENU);
        if (i == FFX_VK_SCALER_OPTION_VARIANT) {
            Cvar_Set("flt_fsr4_variant", variant_index_to_string((int)value->value));
        } else {
            Cvar_SetByVar(Cvar_WeakGet(binding(model, i)), value->string, FROM_MENU);
        }
    }
}

static void scaler_pop(menuFrameWork_t *menu)
{
    /* Widgets use distinct per-model cvars: hidden items cannot overwrite the
     * active model's values, including on pop after several model switches.
     */
    Menu_Pop(menu);
    apply_settings();
    update_description(menu);
}

static bool scaler_push(menuFrameWork_t *menu)
{
    /* Preserve console/config edits to the active engine settings on entry. */
    int model = model_cvar->integer;
    if (ffxVkScalerModel(model)) {
        for (int i = 0; i < FFX_VK_SCALER_OPTION_COUNT; ++i) {
            cvar_t *saved = settings[model][i];
            if (!saved)
                continue;
            if (i == FFX_VK_SCALER_OPTION_VARIANT) {
                cvar_t *effective = Cvar_FindVar("flt_fsr4_variant");
                if (effective)
                    Cvar_SetValue(saved, (float)variant_string_to_index(effective->string), FROM_CODE);
            } else {
                cvar_t *effective = Cvar_FindVar(binding(model, i));
                if (effective && ffxVkScalerOptionValueValid(model, i, effective->value))
                    Cvar_SetByVar(saved, effective->string, FROM_CODE);
            }
        }
    }
    update_description(menu);
    return Menu_Push(menu);
}

static menuSound_t settings_changed(menuCommon_t *item)
{
    scaler_pop(item->parent);
    Menu_Init(item->parent);
    return QMS_SILENT;
}

static void scaler_draw(menuFrameWork_t *menu)
{
    /* A renderer rebuild can change readiness while this page is open. */
    for (int i = 0; i < menu->nitems; ++i) {
        menuCommon_t *item = menu->items[i];
        if (item->visible && item->visible(item) == !!(item->flags & QMF_HIDDEN)) {
            update_description(menu);
            Menu_Init(menu);
            break;
        }
    }
    Menu_Draw(menu);
}

void UI_AddScalerSettings(menuFrameWork_t *menu)
{
    /* The selector must be the first item: this also prevents accidental
     * insertion of two independent settings groups into one menu. */
    if (menu->nitems) {
        Com_WPrintf("scaler_settings must be the first menu control\n");
        return;
    }
    menuSpinControl_t *selector = UI_Mallocz(sizeof(*selector));
    model_cvar = Cvar_Get("flt_upscaler", "0", CVAR_ARCHIVE);
    available_models = Cvar_WeakGet("flt_scaler_available");
    selector->generic.type = MTYPE_PAIRS;
    selector->generic.name = UI_CopyString("model");
    selector->generic.status = UI_CopyString("Choose an upscaler model. Only its applicable settings appear below. Supports AMD FSR 3 / FSR 4.1.2 (RDNA2 INT8), Intel XeSS 2.0.2 (Multi-Preset INT8/DP4a U-Net), and NVIDIA DLSS 4 / d4r Model K.");
    selector->generic.change = settings_changed;
    selector->cvar = model_cvar;
    selector->numItems = FFX_VK_SCALER_COUNT;
    selector->itemnames = UI_Mallocz((FFX_VK_SCALER_COUNT + 1) * sizeof(char *));
    selector->itemvalues = UI_Mallocz((FFX_VK_SCALER_COUNT + 1) * sizeof(char *));
    for (int model = 0; model < FFX_VK_SCALER_COUNT; ++model) {
        const FfxVkScalerModel *m = ffxVkScalerModel(model);
        selector->itemnames[model] = UI_CopyString(m->label);
        selector->itemvalues[model] = UI_CopyString(va("%d", model));
    }
    Menu_AddItem(menu, selector);

    for (int model = 0; model < FFX_VK_SCALER_COUNT; ++model) {
        const FfxVkScalerModel *m = ffxVkScalerModel(model);
        for (int i = 0; i < FFX_VK_SCALER_OPTION_COUNT; ++i) {
            if (!ffxVkScalerHasOption(model, i))
                continue;
            const FfxVkScalerOption *o = ffxVkScalerOption(i);
            char name[128], initial[32];
            Q_snprintf(name, sizeof(name), "ui_scaler_%s_%s", m->key, o->key);
            Q_snprintf(initial, sizeof(initial), "%g", o->defaultValue);
            cvar_t *legacy = Cvar_FindVar(binding(model, i));
            char legacy_val[32] = {0};
            if (i == FFX_VK_SCALER_OPTION_VARIANT && legacy) {
                Q_snprintf(legacy_val, sizeof(legacy_val), "%d",
                           variant_string_to_index(legacy->string));
            }
            /* Seed new preferences from existing engine settings. Cvar_Get
             * retains an already archived per-model value on later loads. */
            settings[model][i] = Cvar_Get(name,
                (i == FFX_VK_SCALER_OPTION_VARIANT && legacy) ? legacy_val :
                (legacy && ffxVkScalerOptionValueValid(model, i, legacy->value) ?
                 legacy->string : initial), CVAR_ARCHIVE);
            menuCommon_t *item;
            if (o->type == FFX_VK_SCALER_OPTION_RANGE) {
                menuSlider_t *s = UI_Mallocz(sizeof(*s));
                item = &s->generic;
                item->type = MTYPE_SLIDER;
                s->cvar = settings[model][i];
                s->minvalue = o->minimum;
                s->maxvalue = o->maximum;
                s->step = o->step;
                s->format = UI_CopyString("%.2f");
            } else {
                menuSpinControl_t *s = UI_Mallocz(sizeof(*s));
                item = &s->generic;
                item->type = MTYPE_PAIRS;
                s->cvar = settings[model][i];
                s->numItems = (int)o->choiceCount;
                s->itemnames = UI_Mallocz((o->choiceCount + 1) * sizeof(char *));
                s->itemvalues = UI_Mallocz((o->choiceCount + 1) * sizeof(char *));
                for (size_t j = 0; j < o->choiceCount; ++j) {
                    s->itemnames[j] = UI_CopyString(o->choices[j].label);
                    s->itemvalues[j] = UI_CopyString(va("%d", o->choices[j].value));
                }
            }
            item->id = model * FFX_VK_SCALER_OPTION_COUNT + i;
            item->name = UI_CopyString(o->label);
            item->status = UI_CopyString(o->description);
            item->visible = option_visible;
            item->change = settings_changed;
            Menu_AddItem(menu, item);
        }
    }
    menuAction_t *controller = UI_Mallocz(sizeof(*controller));
    controller->generic.type = MTYPE_ACTION;
    controller->generic.name = UI_CopyString("dynamic resolution controller...");
    controller->generic.status = UI_CopyString("Configure the engine's dynamic-resolution timing and scale bounds.");
    controller->generic.uiFlags = UI_CENTER;
    controller->generic.visible = controller_visible;
    controller->generic.activate = open_controller;
    Menu_AddItem(menu, controller);
    menu->push = scaler_push;
    menu->pop = scaler_pop;
    menu->draw = scaler_draw;
}
#else
void UI_AddScalerSettings(menuFrameWork_t *menu)
{
    (void)menu;
}
#endif
